import os
from dotenv import load_dotenv
load_dotenv()

# --- Langfuse Integration ---
from langfuse import Langfuse
# Import specific model for chat prompts if needed, or pass messages directly
# from langfuse.model import CreateChatPrompt # Older, might be PromptInput or similar now
# For recent versions, passing the array of messages to 'prompt' in .generation()
# is often supported, but the error suggests a deeper issue with its internal processing.
# Let's try creating a langfuse.ChatPrompt an object if the direct list isn't working.

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

langfuse_client_for_tracing = None
try:
    if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
        langfuse_client_for_tracing = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST
        )
        print("Langfuse client initialized.")
    else:
        print("Warning: Langfuse environment variables not fully set.")
except Exception as e:
    print(f"Error initializing Langfuse client: {e}")

# --- OpenAI Client Initialization ---
from openai import OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_llm_client = None
try:
    if not OPENAI_API_KEY:
        print("Warning: OPENAI_API_KEY environment variable not set.")
    else:
        openai_llm_client = OpenAI(api_key=OPENAI_API_KEY)
        print("OpenAI client initialized.")
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")

# --- FastAPI and other imports ---
from fastapi import FastAPI, HTTPException, Security, Request
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
import chromadb # For RAG
from sentence_transformers import SentenceTransformer # For RAG
from chromadb.utils import embedding_functions # For RAG
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.middleware.cors import CORSMiddleware

# --- App and other configurations ---
EXPECTED_API_KEY = os.getenv("MY_APP_API_KEY")
CHROMA_DB_PATH = "./chroma_db"
RAG_COLLECTION_NAME = "code_documentation_store"
RAG_EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
RAG_NUM_RESULTS = 1
DISTANCE_THRESHOLD = 1.0 # Adjust this after experimentation

chroma_client = None; code_collection = None
try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    st_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=RAG_EMBEDDING_MODEL_NAME)
    code_collection = chroma_client.get_collection(name=RAG_COLLECTION_NAME, embedding_function=st_ef)
    print(f"ChromaDB collection '{RAG_COLLECTION_NAME}' retrieved. Count: {code_collection.count()}")
    if code_collection.count() == 0: print(f"Warning: ChromaDB collection is empty.")
except Exception as e: print(f"Error initializing ChromaDB: {e}")

limiter = Limiter(key_func=get_remote_address, default_limits=["5/minute"])
API_KEY_NAME = "X-API-Key"; api_key_header_auth = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str | None = Security(api_key_header_auth)):
    if not EXPECTED_API_KEY: print("CRITICAL SERVER ERROR: MY_APP_API_KEY is not set."); raise HTTPException(500, "API Key auth not configured.")
    if api_key_header is None: raise HTTPException(403, "Not authenticated: X-API-Key header missing.")
    if api_key_header == EXPECTED_API_KEY: return api_key_header
    else: raise HTTPException(403, "Could not validate credentials")

class CodeInput(BaseModel): code: str; language: str | None = None
class DocumentationOutput(BaseModel): message: str; original_code: str; generated_documentation: str | None = None

app = FastAPI()
origins = [ "http://localhost:3000", "http://localhost:3001", "https://ai-code-doc-generator.vercel.app" ]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["X-API-Key", "Content-Type", "Authorization"])          
app.state.limiter = limiter; app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/")
@limiter.limit("20/minute") 
async def read_root(request: Request): 
    if langfuse_client_for_tracing:
        try: langfuse_client_for_tracing.trace(name="read_root_trace", user_id=request.client.host if request.client else "unknown_client")
        except Exception as e: print(f"Langfuse error in read_root (non-critical): {e}")
    return {"message": "Welcome to the AI Code Documentation Generator API"}

# ... (all imports and initializations as in the previous "complete main.py")
# ... (FastAPI app setup, auth, other endpoints as before) ...

@app.post("/generate-documentation/", response_model=DocumentationOutput, dependencies=[Security(get_api_key)])
@limiter.limit("5/minute") 
async def generate_docs(request: Request, input_data: CodeInput):
    current_trace = None; generation_span = None; rag_retrieval_span = None
    rag_context_content = ""; rag_context_retrieved_count = 0; rag_context_used_in_prompt = False 
    language = input_data.language.lower() if input_data.language else "unknown"
    code_snippet = input_data.code

    if langfuse_client_for_tracing:
        try:
            current_trace = langfuse_client_for_tracing.trace( # Main trace for the request
                name="generate-code-documentation-rag",
                user_id=request.client.host if request.client else "unknown_client",
                metadata={"language": language, "code_length": len(code_snippet), 
                          "rag_context_used_in_prompt": False, "rag_retrieved_count_passing_threshold": 0},
                tags=["core-feature", "rag", f"lang:{language}"])
            current_trace.update(input={"code": code_snippet, "language": language}) # Log overall input
        except Exception as e: print(f"Langfuse error starting trace: {e}")

    # --- RAG Retrieval (as before) ---
    retrieved_docs_details_for_log = []
    if code_collection: 
        if current_trace:
            try: rag_retrieval_span = current_trace.span(name="rag-chromadb-retrieval", input={"query_code_preview": code_snippet[:100] + "..."})
            except Exception as e: print(f"Langfuse error starting RAG span: {e}")
        try:
            results = code_collection.query(query_texts=[code_snippet], n_results=RAG_NUM_RESULTS, include=["documents", "metadatas", "distances"])
            if results and results.get('ids') and results['ids'][0] and results['documents'][0] and results['distances'] and results['distances'][0]:
                temp_rag_context_parts = ["Relevant examples from knowledge base (passed relevance threshold):\n"]
                for i in range(len(results['ids'][0])):
                    current_distance = results['distances'][0][i]
                    if current_distance < DISTANCE_THRESHOLD:
                        retrieved_code = results['documents'][0][i]; retrieved_meta = results['metadatas'][0][i]
                        golden_doc = retrieved_meta.get('golden_doc', 'N/A'); lang_retrieved = retrieved_meta.get('language', 'unknown')
                        temp_rag_context_parts.extend([f"\n--- Example Snippet (lang: {lang_retrieved}, dist: {current_distance:.4f}) ---\n", f"Code:\n```\n{retrieved_code}\n```\n", f"Its Documentation:\n{golden_doc}\n", "--- End Example Snippet ---\n"])
                        retrieved_docs_details_for_log.append({"id": results['ids'][0][i], "distance": current_distance})
                if retrieved_docs_details_for_log: 
                    rag_context_content = "\n".join(temp_rag_context_parts); rag_context_used_in_prompt = True; rag_context_retrieved_count = len(retrieved_docs_details_for_log)
            if rag_retrieval_span:
                try: rag_retrieval_span.end(output={"retrieved_items_count_total": len(results['ids'][0]) if results and results.get('ids') else 0, "retrieved_items_passing_threshold": rag_context_retrieved_count,"threshold_used": DISTANCE_THRESHOLD, "passed_details": retrieved_docs_details_for_log})
                except Exception as e: print(f"Langfuse error ending RAG span: {e}")
        except Exception as e_rag: print(f"Error during RAG retrieval: {e_rag}"); # ... (handle RAG span error) ...
    # --- End RAG ---

    if current_trace: # Update trace metadata with final RAG outcome
        try: current_trace.update(metadata={"language": language, "code_length": len(code_snippet), "rag_context_used_in_prompt": rag_context_used_in_prompt, "rag_retrieved_count_passing_threshold": rag_context_retrieved_count})
        except Exception as e: print(f"Langfuse error updating trace metadata: {e}")

    if not openai_llm_client: # ... (OpenAI client check) ...
        if current_trace: current_trace.update(level="ERROR", status_message="OpenAI client not initialized", output={"error": "OpenAI client misconfiguration"})
        raise HTTPException(503, "AI service client not initialized.")

    # --- Construct Final Prompt (as before) ---
    final_prompt_parts = [] # ... (construct your final_prompt_parts and prompt_to_llm_str as before) ...
    if rag_context_used_in_prompt and rag_context_content: final_prompt_parts.extend(["Consider the following relevant examples...\n", rag_context_content, "\nBased on the examples above...\n", "-" * 20 + "\n"])
    final_prompt_parts.extend(["You are an expert programmer...", f"The language of the code is: {language}.", "Primary code snippet to document:", f"```\n{code_snippet}\n```", "\nGenerated documentation should include:", "1...", "2...", "3..."]) # Ensure full prompt details
    if language == "python": final_prompt_parts.extend(["\nFor Python...", "Example...", "\"\"\"...\"\"\""]) 
    elif language == "javascript": final_prompt_parts.extend(["\nFor JavaScript...", "Example...", "/**...*/"])
    else: final_prompt_parts.extend(["\nFor this language..."])
    final_prompt_parts.append("\nBe precise. Output only the documentation block itself...")
    prompt_to_llm_str = "\n".join(final_prompt_parts)
    # --- End Construct Final Prompt ---
    
    openai_api_messages = [{"role": "system", "content": "You are a precise code documentation generator."}, {"role": "user", "content": prompt_to_llm_str}]

    # --- MODIFIED LANGFUSE GENERATION SPAN CREATION ---
    if current_trace:
        try:
            generation_span = current_trace.generation(
                name="openai-documentation-generation", # Name is important
                model="gpt-3.5-turbo",                 # Model name
                model_parameters={"temperature": 0.2}  # Other parameters
                # We are OMITTING the 'prompt' parameter here at creation
            )
            # We can try to update the prompt later if needed, or rely on OpenAI auto-instrumentation to fill it.
            # For now, let's see if creating the span without the prompt avoids the error.
            # We will log the prompt_to_llm_str as part of the input to this span when we call .end()
            generation_span.update(input=prompt_to_llm_str) # Or input=openai_api_messages

        except Exception as e:
            print(f"Langfuse error starting generation span: {e}")
            generation_span = None
    # --- END MODIFIED LANGFUSE GENERATION SPAN CREATION ---

    try:
        completion_obj = openai_llm_client.chat.completions.create(
            model="gpt-3.5-turbo", 
            messages=openai_api_messages, 
            temperature=0.2
        )
        generated_doc = completion_obj.choices[0].message.content.strip()
        openai_usage_data = completion_obj.usage if hasattr(completion_obj, 'usage') else None
        
        if generation_span:
            try:
                generation_span.end(
                    output=generated_doc, 
                    usage=openai_usage_data
                    # The 'input' was set with .update() after creation
                )
            except Exception as e:
                print(f"Langfuse error ending generation span: {e}")
        
        # ... (Backtick cleanup and final trace update as before) ...
        if generated_doc.startswith("```") and generated_doc.endswith("```"): # ...
            lines = generated_doc.split('\n');
            if len(lines) > 2:
                cleaned_doc_lines = lines[1:-1]
                if cleaned_doc_lines and not (cleaned_doc_lines[0].strip().startswith('"""') or cleaned_doc_lines[0].strip().startswith('/**')): pass
                generated_doc = "\n".join(cleaned_doc_lines).strip()

        response_message = "Documentation generated successfully." # ... (dynamic message logic as before) ...
        if code_collection: 
            if rag_context_used_in_prompt: response_message += " (Relevant RAG context used)."
            else: response_message += " (No highly relevant RAG context found)."

        if current_trace: 
            try: current_trace.update(output={"generated_documentation": generated_doc, "response_message_detail": response_message.split('.')[-1].strip()}, level="DEFAULT")
            except Exception as e: print(f"Langfuse error updating trace output: {e}")
        
        # --- Inline Evaluation (Ensure EVALUATION_SCRIPT_AVAILABLE logic is here) ---
        # ... your evaluation and langfuse.score() logic ...

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