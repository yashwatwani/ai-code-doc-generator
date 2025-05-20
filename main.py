# main.py
import os
import json
import re
from dotenv import load_dotenv

load_dotenv() 

from fastapi import FastAPI, HTTPException, Security, Request
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from openai import OpenAI 
from langfuse import Langfuse
import chromadb
from chromadb.utils import embedding_functions
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.middleware.cors import CORSMiddleware

# --- FastAPI Cache Integration ---
from fastapi_cache import FastAPICache # Main import
from fastapi_cache.backends.inmemory import InMemoryBackend # In-memory cache backend
from fastapi_cache.decorator import cache # The decorator for caching endpoints

# ... (Evaluation Script Import, Configurations, Client Initializations - remain the same) ...
# --- Evaluation Script Import (with fallback) ---
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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EXPECTED_API_KEY = os.getenv("MY_APP_API_KEY") 
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db_local_default") 
RAG_COLLECTION_NAME = os.getenv("RAG_COLLECTION_NAME", "code_documentation_store_default")
RAG_EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
RAG_NUM_RESULTS = 1 
DISTANCE_THRESHOLD = 1.0 
KNOWLEDGE_BASE_FILEPATH = "knowledge_base_data.json"

# --- Initialize Clients (Langfuse, OpenAI, ChromaDB - as before) ---
langfuse_client_for_tracing = None 
openai_llm_client = None
chroma_client = None
code_collection = None
RAG_ENABLED = False 
# Langfuse
try: 
    if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
        langfuse_client_for_tracing = Langfuse(public_key=LANGFUSE_PUBLIC_KEY, secret_key=LANGFUSE_SECRET_KEY, host=LANGFUSE_HOST)
        print("INFO: Langfuse client initialized. OpenAI calls should be auto-instrumented.")
    else: print("WARNING: Langfuse environment variables not fully set. Langfuse tracing will be disabled.")
except Exception as e: print(f"ERROR: Initializing Langfuse client: {e}")
# OpenAI
from openai import OpenAI 
try: 
    if not OPENAI_API_KEY: print("WARNING: OPENAI_API_KEY environment variable not set. OpenAI calls will fail.")
    else: openai_llm_client = OpenAI(api_key=OPENAI_API_KEY); print("INFO: OpenAI client initialized.")
except Exception as e: print(f"ERROR: Initializing OpenAI client: {e}")
# ChromaDB and RAG Population 
try:
    print(f"INFO: Attempting to initialize ChromaDB client with path: {CHROMA_DB_PATH}") # This line was correct
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    # ... (rest of your ChromaDB and RAG init logic as it was) ...
    print(f"INFO: ChromaDB client using path: {CHROMA_DB_PATH}")
    st_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=RAG_EMBEDDING_MODEL_NAME)
    print("INFO: RAG embedding model loaded.")
    code_collection = chroma_client.get_or_create_collection(name=RAG_COLLECTION_NAME,embedding_function=st_ef)
    print(f"INFO: ChromaDB collection '{RAG_COLLECTION_NAME}' ensured. Current count: {code_collection.count()}")
    try:
        with open(KNOWLEDGE_BASE_FILEPATH, 'r') as f: knowledge_base = json.load(f)
        print(f"INFO: Loaded {len(knowledge_base)} items from {KNOWLEDGE_BASE_FILEPATH} for RAG.")
        if code_collection.count() >= len(knowledge_base) and len(knowledge_base) > 0 :
            print(f"INFO: Collection '{RAG_COLlection_NAME}' appears populated. Skipping upsertion.")
            RAG_ENABLED = True
        elif knowledge_base:
            # ... (upsert logic) ...
            documents_to_add, metadatas_to_add, ids_to_add = [], [], []
            for item in knowledge_base:
                doc, lang, golden, item_id = item.get("code_snippet"), item.get("language"), item.get("golden_documentation"), item.get("id")
                if not all([doc, lang, golden, item_id]): continue
                documents_to_add.append(doc); metadatas_to_add.append({"language": lang, "golden_doc": golden}); ids_to_add.append(item_id)
            if ids_to_add:
                code_collection.upsert(documents=documents_to_add, metadatas=metadatas_to_add, ids=ids_to_add)
                print(f"INFO: Upserted docs. New count: {code_collection.count()}")
            RAG_ENABLED = True if code_collection.count() > 0 else False
        else: RAG_ENABLED = True if code_collection.count() > 0 else False # If KB empty but collection had data
    except FileNotFoundError: print(f"ERROR: {KNOWLEDGE_BASE_FILEPATH} not found."); RAG_ENABLED = True if code_collection and code_collection.count() > 0 else False
    except Exception as e_populate: print(f"ERROR: Populating RAG: {e_populate}"); RAG_ENABLED = True if code_collection and code_collection.count() > 0 else False
    if RAG_ENABLED: print(f"INFO: RAG enabled. Collection count: {code_collection.count()}")
    else: print(f"WARNING: RAG NOT enabled or empty. Count: {code_collection.count() if code_collection else 'N/A'}.")
except Exception as e_chroma_init:
    print(f"ERROR: ChromaDB init: {e_chroma_init}. RAG disabled.")
    chroma_client = None; code_collection = None; RAG_ENABLED = False


# --- FastAPI App Instance and Middleware Setup ---
app = FastAPI()
limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"]) 
# CORS
configured_origins = ["http://localhost:3000", "http://localhost:3001", os.getenv("FRONTEND_URL")]
origins = [origin for origin in configured_origins if origin]
if not origins: origins = ["http://localhost:3000", "http://localhost:3001"] # Fallback for local
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["X-API-Key", "Content-Type", "Authorization"])          
# Rate Limiter
app.state.limiter = limiter #type: ignore
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- FastAPI Cache Initialization (should be done once on app startup) ---
@app.on_event("startup")
async def startup_event():
    print("INFO: Initializing FastAPI Cache with InMemoryBackend.")
    FastAPICache.init(InMemoryBackend(), prefix="fastapi-cache")
    print("INFO: FastAPI Cache initialized.")

# ... (API Key Auth, Pydantic Models - remain the same) ...
API_KEY_NAME = "X-API-Key"; api_key_header_auth = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
async def get_api_key(api_key_header: str | None = Security(api_key_header_auth)):
    if api_key_header is None: raise HTTPException(status_code=403, detail="Not authenticated: X-API-Key header missing.")
    if not EXPECTED_API_KEY: print("CRITICAL SERVER ERROR: MY_APP_API_KEY not set."); raise HTTPException(status_code=500, detail="API Key authentication is not configured correctly on the server.")
    if api_key_header == EXPECTED_API_KEY: return api_key_header
    else: raise HTTPException(status_code=403, detail="Could not validate credentials")
class CodeInput(BaseModel): code: str; language: str | None = None
class DocumentationOutput(BaseModel): message: str; original_code: str; generated_documentation: str | None = None

# --- API Endpoints ---
@app.get("/")
@limiter.limit("30/minute") 
async def read_root(request: Request): 
    # ... (Langfuse tracing for root - as before) ...
    if langfuse_client_for_tracing:
        try: langfuse_client_for_tracing.trace(name="read_root_trace", user_id=request.client.host if request.client else "unknown_client")
        except Exception as e: print(f"Langfuse error in read_root (non-critical): {e}")
    return {"message": "Welcome to the AI Code Documentation Generator API"}

# Apply caching to this endpoint
@app.post("/generate-documentation/", response_model=DocumentationOutput, dependencies=[Security(get_api_key)])
@limiter.limit("5/minute") 
@cache(expire=300) # Cache responses for this endpoint for 300 seconds (5 minutes)
async def generate_docs(request: Request, input_data: CodeInput): # Request object needed by @cache if using default key builder with request
    # ... (Your existing Langfuse trace creation, RAG logic, OpenAI call, and evaluation logic) ...
    # This entire block will only execute if the request is not found in cache or is expired.
    # IMPORTANT: The @cache decorator uses the function signature (including input_data) to build a cache key.
    # If CodeInput includes mutable objects or things that don't stringify consistently for the key,
    # you might need a custom key_builder for the @cache decorator. For simple strings like code and language, it's usually fine.

    print(f"INFO: /generate-documentation called (cache miss or expired) for lang: {input_data.language}, code preview: {input_data.code[:50]}...") # Log when not served from cache

    current_trace, generation_span, rag_retrieval_span = None, None, None
    # ... (rest of your generate_docs logic as it was, starting with Langfuse trace creation) ...
    # Ensure all return paths within this function (success and error) are considered by the cache.
    # The cache will store the returned value from this function.
    # Your existing try/except for HTTPException should be fine, as FastAPI Cache
    # typically doesn't cache responses that resulted in server errors unless configured to.

    # --- Start of existing logic from your main.py ---
    rag_context_content, rag_context_used_in_prompt = "", False
    rag_context_retrieved_count, language, code_snippet = 0, input_data.language.lower() if input_data.language else "unknown", input_data.code

    if langfuse_client_for_tracing:
        try:
            current_trace = langfuse_client_for_tracing.trace(
                name="generate-code-documentation-rag",
                user_id=request.client.host if request.client else "unknown_client",
                metadata={ "language": language, "code_length": len(code_snippet), "rag_context_used_in_prompt": False, "rag_retrieved_count_passing_threshold": 0 },
                tags=["core-feature", "rag", f"lang:{language}"]
            )
            current_trace.update(input={"code_preview": code_snippet[:200]+"...", "language": language})
        except Exception as e: print(f"Langfuse error starting trace: {e}")
    
    retrieved_docs_details_for_log = []
    if RAG_ENABLED and code_collection:
        if current_trace:
            try: rag_retrieval_span = current_trace.span(name="rag-chromadb-retrieval", input={"query_code_preview": code_snippet[:100] + "..."})
            except Exception as e: print(f"Langfuse error starting RAG span: {e}")
        try:
            results = code_collection.query(query_texts=[code_snippet], n_results=RAG_NUM_RESULTS, include=["documents", "metadatas", "distances"])
            if results and results.get('ids') and results['ids'][0]:
                temp_rag_context_parts = ["\n\n--- Relevant Examples from Knowledge Base (Passed Relevance Threshold) ---\n"]
                for i in range(len(results['ids'][0])):
                    if not results['documents'][0] or not results['distances'] or not results['distances'][0] : continue
                    current_distance = results['distances'][0][i]
                    if current_distance < DISTANCE_THRESHOLD:
                        retrieved_code, retrieved_meta = results['documents'][0][i], results['metadatas'][0][i]
                        golden_doc, lang_retrieved = retrieved_meta.get('golden_doc', 'N/A'), retrieved_meta.get('language', 'unknown')
                        temp_rag_context_parts.extend([f"\n--- Example (lang: {lang_retrieved}, dist: {current_distance:.4f}) ---\n", f"Code:\n```\n{retrieved_code}\n```\n", f"Its Documentation:\n{golden_doc}\n--- End Example ---\n"])
                        retrieved_docs_details_for_log.append({"id": results['ids'][0][i], "distance": current_distance})
                if len(temp_rag_context_parts) > 1: 
                    rag_context_content = "".join(temp_rag_context_parts)
                    rag_context_used_in_prompt = True
                rag_context_retrieved_count = len(retrieved_docs_details_for_log)
            if not retrieved_docs_details_for_log: print("RAG: No documents found or passed threshold.")
            if rag_retrieval_span:
                try: rag_retrieval_span.end(output={"retrieved_items_count_total": len(results['ids'][0]) if results and results.get('ids') and results['ids'][0] else 0, "retrieved_items_passing_threshold": rag_context_retrieved_count,"threshold_used": DISTANCE_THRESHOLD, "passed_details": retrieved_docs_details_for_log})
                except Exception as e: print(f"Langfuse error ending RAG span: {e}")
        except Exception as e_rag: 
            print(f"Error during RAG retrieval: {e_rag}")
            if rag_retrieval_span:
                try: rag_retrieval_span.end(level="ERROR", status_message=str(e_rag))
                except Exception as e: print(f"Langfuse error ending RAG span with error: {e}")
    else: print("RAG system not enabled or collection unavailable. Skipping retrieval.")

    if current_trace:
        try: current_trace.update(metadata={"language": language, "code_length": len(code_snippet), "rag_context_used_in_prompt": rag_context_used_in_prompt, "rag_retrieved_count_passing_threshold": rag_context_retrieved_count})
        except Exception as e: print(f"Langfuse error updating trace metadata: {e}")

    if not openai_llm_client: 
        if current_trace: current_trace.update(level="ERROR", status_message="OpenAI client not initialized", output={"error": "OpenAI client misconfiguration"})
        raise HTTPException(status_code=503, detail="AI service client not initialized. Check server configuration for OpenAI API key.")

    final_prompt_parts = [] # ... (your prompt building logic as before) ...
    if rag_context_used_in_prompt and rag_context_content: final_prompt_parts.extend(["Consider the following relevant examples...\n", rag_context_content, "\nBased on the examples above...\n", "-" * 20 + "\n"])
    base_prompt_instructions = ["You are an expert programmer...", f"The language of the code is: {language}.", "Primary code snippet to document:", f"```\n{code_snippet}\n```", "\nGenerated documentation should include:", "1...", "2...", "3..."]
    final_prompt_parts.extend(base_prompt_instructions)
    if language == "python": final_prompt_parts.extend(["\nFor Python...", "Example...", "\"\"\"...\"\"\""]) 
    elif language == "javascript": final_prompt_parts.extend(["\nFor JavaScript...", "Example...", "/**...*/"])
    else: final_prompt_parts.extend(["\nFor this language..."])
    final_prompt_parts.append("\nBe precise. Output only the documentation block itself...")
    prompt_to_llm_str = "\n".join(final_prompt_parts)
    openai_api_messages = [{"role": "system", "content": "You are a precise code documentation generator."}, {"role": "user", "content": prompt_to_llm_str}]
    
    if current_trace:
        try: generation_span = current_trace.generation(name="openai-documentation-generation", model="gpt-3.5-turbo",model_parameters={"temperature": 0.2}, input=openai_api_messages) 
        except Exception as e: print(f"Langfuse error starting generation span: {e}"); generation_span = None 
            
    try:
        completion_obj = openai_llm_client.chat.completions.create(model="gpt-3.5-turbo", messages=openai_api_messages, temperature=0.2)
        generated_doc = completion_obj.choices[0].message.content.strip()
        openai_sdk_usage_object = completion_obj.usage if hasattr(completion_obj, 'usage') else None
        if generation_span:
            try: generation_span.end(output=generated_doc, usage=openai_sdk_usage_object)
            except Exception as e: print(f"Langfuse error ending generation span with usage: {e}")
        
        # ... (backtick cleanup as before) ...
        if generated_doc.startswith("```") and generated_doc.endswith("```"):
            lines = generated_doc.split('\n');
            if len(lines) > 1: 
                first_content_line_index = 1
                if re.match(r"^[a-zA-Z0-9_]*$", lines[1].strip()) and len(lines) > 2 : 
                    if not (lines[1].strip().startswith('"""') or lines[1].strip().startswith("'''") or lines[1].strip().startswith('/**')):
                         first_content_line_index = 2
                if len(lines) > first_content_line_index : 
                    cleaned_doc_lines = lines[first_content_line_index : -1] 
                    generated_doc = "\n".join(cleaned_doc_lines).strip()
                else: generated_doc = ""
            else: generated_doc = ""

        response_message = "Documentation generated successfully." # ... (response message logic as before) ...
        if RAG_ENABLED: 
            if rag_context_used_in_prompt: response_message += " (Relevant RAG context used)."
            else: response_message += " (No highly relevant RAG context found)."
        
        if current_trace: 
            try: current_trace.update(output={"generated_documentation": generated_doc, "response_message_detail": response_message}, level="DEFAULT")
            except Exception as e: print(f"Langfuse error updating trace output: {e}")
        
        if EVALUATION_SCRIPT_AVAILABLE and current_trace and generated_doc: # ... (evaluation logic as before) ...
            try:
                eval_results = evaluate_documentation(code_snippet, language, generated_doc)
                for metric_name, metric_value in eval_results.items():
                    if metric_name in ["Code_Analysis_Debug", "Notes"]: continue
                    score_val_for_langfuse = 0.0; comment_for_langfuse = str(metric_value)
                    if isinstance(metric_value, bool): score_val_for_langfuse = 1.0 if metric_value else 0.0
                    elif isinstance(metric_value, (int, float)): score_val_for_langfuse = float(metric_value)
                    elif isinstance(metric_value, str): 
                        if metric_name == "M5_Word_Count_Readability":
                            if metric_value == "Acceptable Length": score_val_for_langfuse = 1.0
                            elif "Short" in metric_value or "Long" in metric_value: score_val_for_langfuse = 0.5 
                        elif "Documentation" in metric_name: 
                            if "Present and Expected" in metric_value or "Absent and Not Expected" in metric_value: score_val_for_langfuse = 1.0
                            elif "Missing but Expected" in metric_value: score_val_for_langfuse = 0.25 
                            elif "Present but Not Expected" in metric_value: score_val_for_langfuse = 0.75 
                    try: 
                        if langfuse_client_for_tracing: current_trace.score(name=metric_name, value=score_val_for_langfuse, comment=comment_for_langfuse)
                    except Exception as e_score: print(f"Langfuse error logging score '{metric_name}': {e_score}")
            except Exception as e_eval_call: print(f"Error calling evaluate_documentation: {e_eval_call}")
            
        return DocumentationOutput(message=response_message, original_code=code_snippet, generated_documentation=generated_doc)

    except HTTPException as http_exc: # ... (HTTPException handling as before) ...
        if generation_span: generation_span.end(level="ERROR", status_message=str(http_exc.detail))
        if current_trace: current_trace.update(level="ERROR", status_message=str(http_exc.detail), output={"error": http_exc.detail})
        raise http_exc
    except Exception as e: # ... (Generic Exception handling as before) ...
        error_message = f"Unhandled error during API processing: {type(e).__name__} - {str(e)}"
        if generation_span: 
            try: generation_span.end(level="ERROR", status_message=error_message)
            except Exception as le_g: print(f"Langfuse error ending gen span with unhandled error: {le_g}")
        if current_trace:  
            try: current_trace.update(level="ERROR", status_message=error_message, output={"error": error_message})
            except Exception as le_t: print(f"Langfuse error updating trace with unhandled error: {le_t}")
        print(error_message) 
        raise HTTPException(status_code=500, detail="An unexpected error occurred during documentation generation.")