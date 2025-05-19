import os
import json
from dotenv import load_dotenv

# Load .env as early as possible - for local dev; Render uses its own env var system
load_dotenv() 

# --- Imports ---
from fastapi import FastAPI, HTTPException, Security, Request
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from openai import OpenAI 
from langfuse import Langfuse
import chromadb
from sentence_transformers import SentenceTransformer 
from chromadb.utils import embedding_functions
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.middleware.cors import CORSMiddleware
import re

# --- Evaluation Script Import (with fallback for Render if not deployed with it) ---
try:
    from evaluate import evaluate_documentation
    EVALUATION_SCRIPT_AVAILABLE = True
    print("INFO: Evaluation script 'evaluate.py' imported successfully.")
except ImportError:
    print("WARNING: Evaluation script 'evaluate.py' not found or error in import. Inline evaluation will be skipped.")
    EVALUATION_SCRIPT_AVAILABLE = False
    def evaluate_documentation(code, language, generated_doc): # Dummy
        print("WARNING: Using dummy evaluate_documentation function (evaluate.py not found/import error).")
        return {}

# --- Configuration ---
# For OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# For this app's own API key auth
EXPECTED_API_KEY = os.getenv("MY_APP_API_KEY") 
# For Langfuse
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
# For RAG with ChromaDB
CHROMA_DB_PATH = "./chroma_db_render"  # Use a distinct path for Render's ephemeral storage
RAG_COLLECTION_NAME = "code_documentation_store_render" # Can be same or different
RAG_EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
RAG_NUM_RESULTS = 1 
DISTANCE_THRESHOLD = 1.0 # Adjust based on experimentation
KNOWLEDGE_BASE_FILEPATH = "knowledge_base_data.json" # Must be in your Git repo

# --- Initialize Clients ---
langfuse_client_for_tracing = None 
openai_llm_client = None
chroma_client = None
code_collection = None
RAG_ENABLED = False # Flag to indicate if RAG setup was successful

# Langfuse
try: 
    if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
        langfuse_client_for_tracing = Langfuse(public_key=LANGFUSE_PUBLIC_KEY, secret_key=LANGFUSE_SECRET_KEY, host=LANGFUSE_HOST)
        print("INFO: Langfuse client initialized.")
    else: print("WARNING: Langfuse environment variables not fully set. Langfuse tracing will be disabled.")
except Exception as e: print(f"ERROR: Initializing Langfuse client: {e}")

# OpenAI
try: 
    if not OPENAI_API_KEY: print("WARNING: OPENAI_API_KEY environment variable not set. OpenAI calls will fail.")
    else: openai_llm_client = OpenAI(api_key=OPENAI_API_KEY); print("INFO: OpenAI client initialized.")
except Exception as e: print(f"ERROR: Initializing OpenAI client: {e}")

# ChromaDB and RAG Population (on startup)
try:
    print(f"INFO: Attempting to initialize ChromaDB client with path: {CHROMA_DB_PATH}")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    print(f"INFO: ChromaDB client using path: {CHROMA_DB_PATH}")
    
    print(f"INFO: Loading RAG embedding model: {RAG_EMBEDDING_MODEL_NAME}...")
    # SentenceTransformer model will be downloaded if not present in Render's environment
    # This might add to startup time on first boot or after cache clear.
    st_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=RAG_EMBEDDING_MODEL_NAME)
    print("INFO: RAG embedding model loaded.")
    
    print(f"INFO: Getting or creating ChromaDB collection: {RAG_COLLECTION_NAME}")
    code_collection = chroma_client.get_or_create_collection(
        name=RAG_COLLECTION_NAME,
        embedding_function=st_ef
    )
    print(f"INFO: ChromaDB collection '{RAG_COLLECTION_NAME}' ensured.")

    # Load knowledge base data and upsert into collection
    try:
        with open(KNOWLEDGE_BASE_FILEPATH, 'r') as f:
            knowledge_base = json.load(f)
        print(f"INFO: Loaded {len(knowledge_base)} items from {KNOWLEDGE_BASE_FILEPATH} for RAG.")

        if knowledge_base:
            documents_to_add = []
            metadatas_to_add = []
            ids_to_add = []
            
            for item in knowledge_base: # Ensure keys match your JSON file exactly
                doc = item.get("code_snippet")
                lang = item.get("language")
                golden = item.get("golden_documentation")
                item_id = item.get("id")

                if not all([doc, lang, golden, item_id]):
                    print(f"WARNING: Skipping item due to missing fields: {item.get('id', 'Unknown ID')}")
                    continue
                documents_to_add.append(doc)
                metadatas_to_add.append({"language": lang, "golden_doc": golden})
                ids_to_add.append(item_id)

            if ids_to_add: # Check if there are valid items to add
                print(f"INFO: Upserting {len(ids_to_add)} documents into ChromaDB collection '{RAG_COLLECTION_NAME}'...")
                code_collection.upsert(documents=documents_to_add, metadatas=metadatas_to_add, ids=ids_to_add)
                print(f"INFO: Upserted documents. Collection count: {code_collection.count()}")
                RAG_ENABLED = True if code_collection.count() > 0 else False
            else:
                print("WARNING: No valid documents to upsert after filtering knowledge base.")
                RAG_ENABLED = True if code_collection.count() > 0 else False # Check existing count
        else:
            print("WARNING: Knowledge base file is empty. No documents to upsert for RAG.")
            RAG_ENABLED = True if code_collection.count() > 0 else False
            
    except FileNotFoundError:
        print(f"ERROR: {KNOWLEDGE_BASE_FILEPATH} not found. RAG cannot be populated.")
        RAG_ENABLED = False
    except Exception as e_populate:
        print(f"ERROR: During RAG data population from JSON: {e_populate}")
        RAG_ENABLED = False

    if RAG_ENABLED: print(f"INFO: RAG system enabled. Collection has {code_collection.count()} documents.")
    else: print(f"WARNING: RAG system NOT enabled or collection is empty. Collection count: {code_collection.count() if code_collection else 'N/A'}.")

except Exception as e:
    print(f"ERROR: FATAL Error initializing ChromaDB components: {e}. RAG will be disabled.")
    chroma_client = None; code_collection = None; RAG_ENABLED = False
# --- End ChromaDB Init ---

# --- FastAPI Setup (as before) ---
limiter = Limiter(key_func=get_remote_address, default_limits=["5/minute"])
API_KEY_NAME = "X-API-Key"; api_key_header_auth = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
async def get_api_key(api_key_header: str | None = Security(api_key_header_auth)): # ... (same as before) ...
    if not EXPECTED_API_KEY: print("CRITICAL SERVER ERROR: MY_APP_API_KEY is not set."); raise HTTPException(500, "API Key auth not configured.")
    if api_key_header is None: raise HTTPException(403, "Not authenticated: X-API-Key header missing.")
    if api_key_header == EXPECTED_API_KEY: return api_key_header
    else: raise HTTPException(403, "Could not validate credentials")
class CodeInput(BaseModel): code: str; language: str | None = None # ... (same as before) ...
class DocumentationOutput(BaseModel): message: str; original_code: str; generated_documentation: str | None = None # ... (same as before) ...
app = FastAPI() # ... (CORS, Rate Limiter setup as before) ...
origins = [ "http://localhost:3000", "http://localhost:3001", "https://ai-code-doc-generator.vercel.app" ]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["X-API-Key", "Content-Type", "Authorization"])          
app.state.limiter = limiter; app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/") # ... (Root endpoint as before) ...
@limiter.limit("20/minute") 
async def read_root(request: Request): 
    if langfuse_client_for_tracing:
        try: langfuse_client_for_tracing.trace(name="read_root_trace", user_id=request.client.host if request.client else "unknown_client")
        except Exception as e: print(f"Langfuse error in read_root (non-critical): {e}")
    return {"message": "Welcome to the AI Code Documentation Generator API"}

@app.post("/generate-documentation/", response_model=DocumentationOutput, dependencies=[Security(get_api_key)])
@limiter.limit("5/minute") 
async def generate_docs(request: Request, input_data: CodeInput):
    # ... (trace, RAG context vars init as before) ...
    current_trace = None; generation_span = None; rag_retrieval_span = None
    rag_context_content = ""; rag_context_retrieved_count = 0; rag_context_used_in_prompt = False 
    language = input_data.language.lower() if input_data.language else "unknown"
    code_snippet = input_data.code

    if langfuse_client_for_tracing: # ... (Start main Langfuse trace - as before) ...
        try:
            current_trace = langfuse_client_for_tracing.trace(name="generate-code-documentation-rag",user_id=request.client.host if request.client else "unknown_client",metadata={"language": language, "code_length": len(code_snippet), "rag_context_used_in_prompt": False, "rag_retrieved_count_passing_threshold": 0},tags=["core-feature", "rag", f"lang:{language}"])
            current_trace.update(input={"code": code_snippet, "language": language})
        except Exception as e: print(f"Langfuse error starting trace: {e}")
    
    # --- RAG: Retrieve Similar Code Snippets ---
    retrieved_docs_details_for_log = []
    if RAG_ENABLED and code_collection:  # Check RAG_ENABLED flag
        # ... (Full RAG logic with distance thresholding - as before) ...
        if current_trace:
            try: rag_retrieval_span = current_trace.span(name="rag-chromadb-retrieval", input={"query_code_preview": code_snippet[:100] + "..."})
            except Exception as e: print(f"Langfuse error starting RAG span: {e}")
        try:
            # print(f"RAG: Querying collection for code: {code_snippet[:50]}...") # Optional debug
            results = code_collection.query(query_texts=[code_snippet], n_results=RAG_NUM_RESULTS, include=["documents", "metadatas", "distances"])
            if results and results.get('ids') and results['ids'][0] and results['documents'][0] and results['distances'] and results['distances'][0]:
                temp_rag_context_parts = ["Relevant examples from knowledge base (passed relevance threshold):\n"]
                for i in range(len(results['ids'][0])):
                    current_distance = results['distances'][0][i]
                    # print(f"RAG: Candidate ID: {results['ids'][0][i]}, Distance: {current_distance}") # Optional debug
                    if current_distance < DISTANCE_THRESHOLD:
                        # print(f"RAG: Candidate ID {results['ids'][0][i]} PASSED threshold.") # Optional debug
                        retrieved_code = results['documents'][0][i]; retrieved_meta = results['metadatas'][0][i]
                        golden_doc = retrieved_meta.get('golden_doc', 'N/A'); lang_retrieved = retrieved_meta.get('language', 'unknown')
                        temp_rag_context_parts.extend([f"\n--- Example Snippet (lang: {lang_retrieved}, dist: {current_distance:.4f}) ---\n", f"Code:\n```\n{retrieved_code}\n```\n", f"Its Documentation:\n{golden_doc}\n", "--- End Example Snippet ---\n"])
                        retrieved_docs_details_for_log.append({"id": results['ids'][0][i], "distance": current_distance})
                if retrieved_docs_details_for_log: 
                    rag_context_content = "\n".join(temp_rag_context_parts); rag_context_used_in_prompt = True; rag_context_retrieved_count = len(retrieved_docs_details_for_log)
            # ... (else print messages for no docs found / no docs passing threshold - as before) ...
            if not retrieved_docs_details_for_log : print("RAG: No documents found or passed threshold.")

            if rag_retrieval_span:
                try: rag_retrieval_span.end(output={"retrieved_items_count_total": len(results['ids'][0]) if results and results.get('ids') and results['ids'][0] else 0, "retrieved_items_passing_threshold": rag_context_retrieved_count,"threshold_used": DISTANCE_THRESHOLD, "passed_details": retrieved_docs_details_for_log})
                except Exception as e: print(f"Langfuse error ending RAG span: {e}")
        except Exception as e_rag: print(f"Error during RAG retrieval: {e_rag}"); # ... (handle RAG span error) ...
    else: print("RAG system not enabled or collection unavailable. Skipping retrieval.")
    # --- End RAG ---

    # ... (Update trace metadata, OpenAI client check, Construct Final Prompt, Langfuse generation span, OpenAI call - as before) ...
    if current_trace: # Update trace metadata with final RAG outcome
        try: current_trace.update(metadata={"language": language, "code_length": len(code_snippet), "rag_context_used_in_prompt": rag_context_used_in_prompt, "rag_retrieved_count_passing_threshold": rag_context_retrieved_count})
        except Exception as e: print(f"Langfuse error updating trace metadata: {e}")
    if not openai_llm_client: 
        if current_trace: current_trace.update(level="ERROR", status_message="OpenAI client not initialized", output={"error": "OpenAI client misconfiguration"})
        raise HTTPException(503, "AI service client not initialized.")
    final_prompt_parts = []
    if rag_context_used_in_prompt and rag_context_content: final_prompt_parts.extend(["Consider the following relevant examples...\n", rag_context_content, "\nBased on the examples above...\n", "-" * 20 + "\n"])
    final_prompt_parts.extend(["You are an expert programmer...", f"The language of the code is: {language}.", "Primary code snippet to document:", f"```\n{code_snippet}\n```", "\nGenerated documentation should include:", "1...", "2...", "3..."])
    if language == "python": final_prompt_parts.extend(["\nFor Python...", "Example...", "\"\"\"...\"\"\""]) 
    elif language == "javascript": final_prompt_parts.extend(["\nFor JavaScript...", "Example...", "/**...*/"])
    else: final_prompt_parts.extend(["\nFor this language..."])
    final_prompt_parts.append("\nBe precise. Output only the documentation block itself...")
    prompt_to_llm_str = "\n".join(final_prompt_parts)
    openai_api_messages = [{"role": "system", "content": "You are a precise code documentation generator."}, {"role": "user", "content": prompt_to_llm_str}]
    if current_trace:
        try: generation_span = current_trace.generation(name="openai-documentation-generation", model="gpt-3.5-turbo",model_parameters={"temperature": 0.2}, prompt=openai_api_messages) 
        except Exception as e: print(f"Langfuse error starting generation span: {e}"); generation_span = None
    try:
        completion_obj = openai_llm_client.chat.completions.create(model="gpt-3.5-turbo", messages=openai_api_messages, temperature=0.2)
        generated_doc = completion_obj.choices[0].message.content.strip()
        openai_usage_data = completion_obj.usage if hasattr(completion_obj, 'usage') else None
        if generation_span:
            try: generation_span.end(output=generated_doc, usage=openai_usage_data)
            except Exception as e: print(f"Langfuse error ending generation span: {e}")
        # Backtick cleanup
        if generated_doc.startswith("```") and generated_doc.endswith("```"):
            lines = generated_doc.split('\n');
            if len(lines) > 2:
                cleaned_doc_lines = lines[1:-1]
                if cleaned_doc_lines and re.match(r"^[a-zA-Z0-9_]*$", cleaned_doc_lines[0].strip()) and not (cleaned_doc_lines[0].strip().startswith('"""') or cleaned_doc_lines[0].strip().startswith('/**')):
                    cleaned_doc_lines = cleaned_doc_lines[1:]
                generated_doc = "\n".join(cleaned_doc_lines).strip()
        
        response_message = "Documentation generated successfully."
        if RAG_ENABLED: 
            if rag_context_used_in_prompt: response_message += " (Relevant RAG context used)."
            else: response_message += " (No highly relevant RAG context found)."
        if current_trace: 
            try: current_trace.update(output={"generated_documentation": generated_doc, "response_message_detail": response_message.split('.')[-1].strip()}, level="DEFAULT")
            except Exception as e: print(f"Langfuse error updating trace output: {e}")
        if EVALUATION_SCRIPT_AVAILABLE and current_trace and generated_doc: # ... (Inline evaluation logic as before) ...
            try:
                eval_results = evaluate_documentation(code_snippet, language, generated_doc)
                for metric_name, metric_value in eval_results.items():
                    if metric_name in ["Code_Analysis_Debug", "Notes"]: continue
                    score_val_for_langfuse = 0; comment_for_langfuse = str(metric_value)
                    if isinstance(metric_value, bool): score_val_for_langfuse = 1 if metric_value else 0
                    elif isinstance(metric_value, str): 
                        if metric_name == "M5_Word_Count_Readability":
                            if metric_value == "Acceptable Length": score_val_for_langfuse = 1
                            elif "Short" in metric_value: score_val_for_langfuse = 0.5 
                            elif "Long" in metric_value: score_val_for_langfuse = 0.5
                        elif "Documentation" in metric_name: 
                            if "Present and Expected" in metric_value or "Absent and Not Expected" in metric_value: score_val_for_langfuse = 1
                            elif "Missing but Expected" in metric_value: score_val_for_langfuse = 0.25
                            elif "Present but Not Expected" in metric_value: score_val_for_langfuse = 0.75
                    try: current_trace.score(name=metric_name, value=score_val_for_langfuse, comment=comment_for_langfuse)
                    except Exception as e_score: print(f"Langfuse error logging score '{metric_name}': {e_score}")
            except Exception as e_eval_call: print(f"Error calling evaluate_documentation: {e_eval_call}")
        return DocumentationOutput(message=response_message, original_code=code_snippet, generated_documentation=generated_doc)
    except Exception as e: # ... (Error handling as before) ...
        if generation_span: 
            try: generation_span.end(level="ERROR", status_message=str(e))
            except Exception as le: print(f"Langfuse error ending gen span with error: {le}")
        if current_trace:  
            try: current_trace.update(level="ERROR", status_message=str(e), output={"error": str(e)})
            except Exception as le: print(f"Langfuse error updating trace with error: {le}")
        print(f"Error during API processing: {type(e).__name__} - {str(e)}") 
        if isinstance(e, HTTPException): raise
        else: raise HTTPException(503, "AI service unavailable or encountered an error during generation.")