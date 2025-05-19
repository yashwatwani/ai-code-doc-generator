import os
from dotenv import load_dotenv
load_dotenv() 

# --- ChromaDB and SentenceTransformer for RAG ---
import chromadb
from sentence_transformers import SentenceTransformer 
from chromadb.utils import embedding_functions

# --- Other Imports (Langfuse, OpenAI, FastAPI, etc.) ---
from langfuse import Langfuse
from openai import OpenAI 
from fastapi import FastAPI, HTTPException, Security, Request
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.middleware.cors import CORSMiddleware

# --- Evaluation Script Import (with fallback) ---
try:
    from evaluate import evaluate_documentation
    EVALUATION_SCRIPT_AVAILABLE = True
    print("Evaluation script 'evaluate.py' imported successfully.")
except ImportError:
    print("Warning: Evaluation script 'evaluate.py' not found or contains errors. Inline evaluation will be skipped.")
    EVALUATION_SCRIPT_AVAILABLE = False
    def evaluate_documentation(code, language, generated_doc): # Dummy function
        print("Warning: Using dummy evaluate_documentation function (evaluate.py not found).")
        return {}


# --- Configuration (including RAG) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EXPECTED_API_KEY = os.getenv("MY_APP_API_KEY") 
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

CHROMA_DB_PATH = "./chroma_db"
RAG_COLLECTION_NAME = "code_documentation_store"
RAG_EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
RAG_NUM_RESULTS = 1 
DISTANCE_THRESHOLD = 1.0 # Example value, adjust based on experiments

# --- Initialize Clients ---
langfuse_client_for_tracing = None 
try: 
    if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
        langfuse_client_for_tracing = Langfuse(public_key=LANGFUSE_PUBLIC_KEY, secret_key=LANGFUSE_SECRET_KEY, host=LANGFUSE_HOST)
        print("Langfuse client initialized.")
    else: print("Warning: Langfuse environment variables not fully set for Langfuse.")
except Exception as e: print(f"Error initializing Langfuse client: {e}")

openai_llm_client = None 
try: 
    if not OPENAI_API_KEY: print("Warning: OPENAI_API_KEY environment variable not set.")
    else:
        openai_llm_client = OpenAI(api_key=OPENAI_API_KEY)
        print("OpenAI client initialized.")
except Exception as e: print(f"Error initializing OpenAI client: {e}")

chroma_client = None; code_collection = None
try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    st_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=RAG_EMBEDDING_MODEL_NAME)
    code_collection = chroma_client.get_collection(name=RAG_COLLECTION_NAME, embedding_function=st_ef)
    print(f"ChromaDB collection '{RAG_COLLECTION_NAME}' retrieved. Count: {code_collection.count()}")
    if code_collection.count() == 0: print(f"Warning: ChromaDB collection '{RAG_COLLECTION_NAME}' is empty.")
except Exception as e: print(f"Error initializing ChromaDB: {e}")

# --- FastAPI Setup ---
limiter = Limiter(key_func=get_remote_address, default_limits=["5/minute"])
API_KEY_NAME = "X-API-Key"; api_key_header_auth = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str | None = Security(api_key_header_auth)):
    if not EXPECTED_API_KEY: 
        print("CRITICAL SERVER ERROR: MY_APP_API_KEY is not set.")
        raise HTTPException(status_code=500, detail="API Key authentication is not configured correctly on the server.")
    if api_key_header is None:
        raise HTTPException(status_code=403, detail="Not authenticated: X-API-Key header missing.")
    if api_key_header == EXPECTED_API_KEY: 
        return api_key_header
    else: 
        raise HTTPException(status_code=403, detail="Could not validate credentials")

class CodeInput(BaseModel): code: str; language: str | None = None
class DocumentationOutput(BaseModel): message: str; original_code: str; generated_documentation: str | None = None

app = FastAPI()
origins = [ 
    "http://localhost:3000", 
    "http://localhost:3001",
    "https://ai-code-doc-generator.vercel.app" # Your deployed Vercel frontend URL
]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, 
                   allow_methods=["GET", "POST", "OPTIONS"], 
                   allow_headers=["X-API-Key", "Content-Type", "Authorization"])          
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- API Endpoints ---
@app.get("/")
@limiter.limit("20/minute") 
async def read_root(request: Request): 
    if langfuse_client_for_tracing:
        try: langfuse_client_for_tracing.trace(name="read_root_trace", user_id=request.client.host if request.client else "unknown_client")
        except Exception as e: print(f"Langfuse error in read_root (non-critical): {e}")
    return {"message": "Welcome to the AI Code Documentation Generator API"}

@app.post("/generate-documentation/", response_model=DocumentationOutput, dependencies=[Security(get_api_key)])
@limiter.limit("5/minute") 
async def generate_docs(request: Request, input_data: CodeInput):
    current_trace = None; generation_span = None; rag_retrieval_span = None
    rag_context_content = ""; rag_context_retrieved_count = 0; rag_context_used_in_prompt = False 

    language = input_data.language.lower() if input_data.language else "unknown"
    code_snippet = input_data.code

    if langfuse_client_for_tracing:
        try:
            current_trace = langfuse_client_for_tracing.trace(
                name="generate-code-documentation-rag",
                user_id=request.client.host if request.client else "unknown_client",
                metadata={"language": language, "code_length": len(code_snippet), "rag_context_used_in_prompt": False, "rag_retrieved_count_passing_threshold": 0},
                tags=["core-feature", "rag", f"lang:{language}"])
            current_trace.update(input={"code": code_snippet, "language": language})
        except Exception as e: print(f"Langfuse error starting trace: {e}")

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
        except Exception as e_rag: print(f"Error during RAG retrieval: {e_rag}") # ... (handle RAG span error) ...
    
    if current_trace: # Update trace metadata with final RAG outcome
        try: current_trace.update(metadata={"language": language, "code_length": len(code_snippet), "rag_context_used_in_prompt": rag_context_used_in_prompt, "rag_retrieved_count_passing_threshold": rag_context_retrieved_count})
        except Exception as e: print(f"Langfuse error updating trace metadata: {e}")

    if not openai_llm_client: 
        if current_trace: current_trace.update(level="ERROR", status_message="OpenAI client not initialized", output={"error": "OpenAI client misconfiguration"})
        raise HTTPException(status_code=503, detail="AI service client not initialized. Check server configuration for OpenAI API key.")

    # --- START OF MODIFIED PROMPT SECTION ---
    final_prompt_parts = []
    if rag_context_used_in_prompt and rag_context_content:
        final_prompt_parts.append("Consider the following relevant examples from a knowledge base to help you generate accurate and stylistically appropriate documentation:\n")
        final_prompt_parts.append(rag_context_content)
        final_prompt_parts.append("\nBased on the examples above, and your general knowledge, generate documentation for the primary code snippet below.\n")
        final_prompt_parts.append("-" * 20 + "\n")
    
    final_prompt_parts.extend([
        "You are an expert programmer tasked with generating high-quality, structured documentation for the following code.",
        f"The language of the code is: {language}.",
        "Primary code snippet to document:",
        f"```\n{code_snippet}\n```",
        "\nGenerated documentation MUST include:",
        "1. A concise summary of what the function/code does.",
        "2. Parameters: For EACH parameter identified in the function signature of the primary code snippet, provide its name, its inferred type (e.g., str, int, list, dict, any, etc.), and a clear description of its purpose. If the function takes no parameters, explicitly state 'No parameters.' under an 'Args:' section for Python, or omit @param tags for JavaScript.",
        "3. Returns: A description of what the function/code returns, including its inferred type. If the function does not return anything explicitly (e.g., a Python function with no return statement or only side effects, or a JavaScript function with no return), state 'None' for Python or omit @returns for JavaScript."
    ])

    if language == "python":
        final_prompt_parts.extend([
            "\nFor Python, the entire documentation MUST be a single, valid PEP 257 multiline docstring starting with \"\"\" and ending with \"\"\".",
            "The 'Args:' section should list each parameter on a new line: `    parameter_name (type): description`.",
            "If there are no parameters, the 'Args:' section should contain only the line: `    None`.",
            "The 'Returns:' section should state the type and description: `    return_type: description`.",
            "If there is no return value, the 'Returns:' section should contain only the line: `    None`.",
            "Example of a complete Python docstring for a function with parameters and a return value:",
            "\"\"\"",
            "Summary of the function.",
            "",
            "Args:",
            "    param1 (str): Description of the first parameter.",
            "    param2 (int, optional): Description of the second. Defaults to 0.",
            "",
            "Returns:",
            "    bool: Description of the boolean return value.",
            "\"\"\""
        ])
    elif language == "javascript":
        final_prompt_parts.extend([
            "\nFor JavaScript, the entire documentation MUST be a JSDoc comment block starting with /** and ending with */.",
            "Use an '@param {type} parameter_name - description' tag for EACH parameter.",
            "If there are no parameters, do not include any '@param' tags.",
            "Use an '@returns {type} description' tag if the function returns a value.",
            "If there is no return value, do not include an '@returns' tag.",
            "Example of a complete JSDoc block for a function with parameters and a return value:",
            "/**",
            " * Summary of the function.",
            " *",
            " * @param {string} param1 - Description of the first parameter.",
            " * @param {number} [optionalParam2] - Description of the optional second parameter.",
            " * @returns {boolean} Description of the boolean return value.",
            " */"
        ])
    else: # General fallback for other languages
        final_prompt_parts.extend([
            "\nFor this language, use a standard block comment style.",
            "Clearly list and describe all parameters (name, type, description) and return values (type, description)."
        ])
    
    final_prompt_parts.append("\nBe precise and thorough. Output ONLY the documentation block itself, without any of your own conversational text, introductions, explanations, or markdown formatting like ``` before or after the documentation block.")
    prompt_to_llm_str = "\n".join(final_prompt_parts)
    # --- END OF MODIFIED PROMPT SECTION ---
    
    openai_api_messages = [{"role": "system", "content": "You are a precise code documentation generator."}, {"role": "user", "content": prompt_to_llm_str}]

    if current_trace:
        try:
            generation_span = current_trace.generation(name="openai-documentation-generation", model="gpt-3.5-turbo",
                                                   model_parameters={"temperature": 0.2}, 
                                                   prompt=openai_api_messages) 
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
                # Further check if the first line of cleaned_doc_lines is a language specifier
                if cleaned_doc_lines and re.match(r"^[a-zA-Z0-9_]*$", cleaned_doc_lines[0].strip()) and not (cleaned_doc_lines[0].strip().startswith('"""') or cleaned_doc_lines[0].strip().startswith('/**')):
                    cleaned_doc_lines = cleaned_doc_lines[1:] # remove language specifier line
                generated_doc = "\n".join(cleaned_doc_lines).strip()
        
        response_message = "Documentation generated successfully."
        if code_collection: 
            if rag_context_used_in_prompt: response_message += " (Relevant RAG context used)."
            else: response_message += " (No highly relevant RAG context found)."

        if current_trace: 
            try: current_trace.update(output={"generated_documentation": generated_doc, "response_message_detail": response_message.split('.')[-1].strip()}, level="DEFAULT")
            except Exception as e: print(f"Langfuse error updating trace output: {e}")
        
        # --- Inline Evaluation ---
        if EVALUATION_SCRIPT_AVAILABLE and current_trace and generated_doc:
            # print(f"Running inline evaluation for trace ID: {current_trace.id}") # Optional debug
            try:
                eval_results = evaluate_documentation(code_snippet, language, generated_doc)
                # print(f"Evaluation results: {eval_results}") # Optional debug
                for metric_name, metric_value in eval_results.items():
                    if metric_name in ["Code_Analysis_Debug", "Notes"]: continue
                    score_val_for_langfuse = 0 
                    comment_for_langfuse = str(metric_value)
                    if isinstance(metric_value, bool): score_val_for_langfuse = 1 if metric_value else 0
                    elif isinstance(metric_value, str):
                        if metric_name == "M5_Word_Count_Readability":
                            if metric_value == "Acceptable Length": score_val_for_langfuse = 1
                            elif metric_value == "Too Short": score_val_for_langfuse = 0.5 
                            elif metric_value == "Too Long": score_val_for_langfuse = 0.5
                        elif "Documentation" in metric_name: 
                            if "Present and Expected" in metric_value: score_val_for_langfuse = 1
                            elif "Absent and Not Expected" in metric_value: score_val_for_langfuse = 1
                            elif "Missing but Expected" in metric_value: score_val_for_langfuse = 0.25
                            elif "Present but Not Expected" in metric_value: score_val_for_langfuse = 0.75
                    try:
                        # print(f"Logging score to Langfuse: Name='{metric_name}', Value={score_val_for_langfuse}") # Optional debug
                        current_trace.score(name=metric_name, value=score_val_for_langfuse, comment=comment_for_langfuse)
                    except Exception as e_score: print(f"Langfuse error logging score '{metric_name}': {e_score}")
            except Exception as e_eval_call: print(f"Error calling evaluate_documentation: {e_eval_call}")
        # --- End Inline Evaluation ---

        return DocumentationOutput(message=response_message, original_code=code_snippet, generated_documentation=generated_doc)
    except Exception as e: 
        if generation_span: 
            try: generation_span.end(level="ERROR", status_message=str(e))
            except Exception as le: print(f"Langfuse error ending gen span with error: {le}")
        if current_trace:  
            try: current_trace.update(level="ERROR", status_message=str(e), output={"error": str(e)})
            except Exception as le: print(f"Langfuse error updating trace with error: {le}")
        print(f"Error during API processing: {type(e).__name__} - {str(e)}") 
        if isinstance(e, HTTPException): raise
        else: raise HTTPException(status_code=503, detail="AI service unavailable or encountered an error during generation.")