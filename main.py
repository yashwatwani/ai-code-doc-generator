import os
from dotenv import load_dotenv

# Load .env as early as possible
load_dotenv() 

# --- Langfuse Integration ---
from langfuse import Langfuse

# --- Evaluation Script Import ---
# Assuming evaluate.py is in the same directory or accessible in PYTHONPATH
try:
    from evaluate import evaluate_documentation
    EVALUATION_SCRIPT_AVAILABLE = True
    print("Evaluation script 'evaluate.py' imported successfully.")
except ImportError:
    print("Warning: Evaluation script 'evaluate.py' not found or contains errors. Inline evaluation will be skipped.")
    EVALUATION_SCRIPT_AVAILABLE = False
    def evaluate_documentation(code, language, generated_doc): # Dummy function
        print("Warning: Using dummy evaluate_documentation function.")
        return {}


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
        print("Warning: Langfuse environment variables not fully set. Langfuse tracing will be disabled.")
except Exception as e:
    print(f"Error initializing Langfuse client: {type(e).__name__} - {e}")

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
    print(f"Error initializing OpenAI client: {type(e).__name__} - {e}")

# --- FastAPI and other imports ---
from fastapi import FastAPI, HTTPException, Security, Request
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.middleware.cors import CORSMiddleware

# --- App and other configurations ---
EXPECTED_API_KEY = os.getenv("MY_APP_API_KEY") 
limiter = Limiter(key_func=get_remote_address, default_limits=["5/minute"])
API_KEY_NAME = "X-API-Key"
api_key_header_auth = APIKeyHeader(name=API_KEY_NAME, auto_error=False) 

async def get_api_key(api_key_header: str | None = Security(api_key_header_auth)):
    if not EXPECTED_API_KEY: 
        print("CRITICAL SERVER ERROR: MY_APP_API_KEY is not set.")
        raise HTTPException(status_code=500, detail="API Key authentication is not configured correctly on the server.")
    if api_key_header is None:
        raise HTTPException(status_code=403, detail="Not authenticated: X-API-Key header missing.")
    if api_key_header == EXPECTED_API_KEY: return api_key_header
    else: raise HTTPException(status_code=403, detail="Could not validate credentials")

class CodeInput(BaseModel): code: str; language: str | None = None
class DocumentationOutput(BaseModel): message: str; original_code: str; generated_documentation: str | None = None

app = FastAPI()
origins = [ 
    "http://localhost:3000", "http://localhost:3001",
    "https://ai-code-doc-generator.vercel.app" # Your deployed frontend URL
]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["X-API-Key", "Content-Type", "Authorization"])          
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
    current_trace = None
    generation_span = None

    if langfuse_client_for_tracing:
        try:
            current_trace = langfuse_client_for_tracing.trace(
                name="generate-code-documentation",
                user_id=request.client.host if request.client else "unknown_client",
                metadata={"language": input_data.language, "code_length": len(input_data.code)},
                tags=["core-feature", f"lang:{input_data.language or 'unknown'}"]
            )
            current_trace.update(input={"code": input_data.code, "language": input_data.language})
        except Exception as e: print(f"Langfuse error starting trace (non-critical): {e}")

    if not openai_llm_client: 
        if current_trace: current_trace.update(level="ERROR", status_message="OpenAI client not initialized", output={"error": "OpenAI client misconfiguration"})
        raise HTTPException(status_code=503, detail="AI service client not initialized. Check server configuration for OpenAI API key.")

    language = input_data.language.lower() if input_data.language else "unknown"
    code_snippet = input_data.code
    
    # --- Construct Prompt (Ensure your full prompt logic is here) ---
    prompt_parts = [
        "You are an expert programmer tasked with generating high-quality, structured documentation for code.",
        f"The language of the code is: {language}.", "Analyze the following code snippet:",
        f"```\n{code_snippet}\n```", "\nGenerate documentation that includes:",
        "1. A concise summary...", "2. A description of its parameters...", "3. A description of what it returns..."
    ] # Abridged for brevity - use your full prompt_parts logic
    if language == "python": prompt_parts.extend(["\nFor Python...", "Example...", "\"\"\"...\"\"\""])
    elif language == "javascript": prompt_parts.extend(["\nFor JavaScript...", "Example...", "/**...*/"])
    else: prompt_parts.extend(["\nFor this language..."])
    prompt_parts.append("\nBe precise...")
    prompt = "\n".join(prompt_parts)
    # --- End Construct Prompt ---

    if current_trace:
        try:
            generation_span = current_trace.generation(name="openai-documentation-generation", model="gpt-3.5-turbo",
                                                   model_parameters={"temperature": 0.2}, prompt=prompt)
        except Exception as e: print(f"Langfuse error starting generation span (non-critical): {e}"); generation_span = None

    try:
        completion_obj = openai_llm_client.chat.completions.create( 
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a precise code documentation generator."}, {"role": "user", "content": prompt}],
            temperature=0.2
        )
        generated_doc = completion_obj.choices[0].message.content.strip()
        openai_usage_data = completion_obj.usage if hasattr(completion_obj, 'usage') else None
        
        if generation_span:
            try: generation_span.end(output=generated_doc, usage=openai_usage_data)
            except Exception as e: print(f"Langfuse error ending generation span (non-critical): {e}")
        
        # Backtick cleanup
        if generated_doc.startswith("```") and generated_doc.endswith("```"): # ... (your cleanup logic) ...
            lines = generated_doc.split('\n');
            if len(lines) > 2:
                cleaned_doc_lines = lines[1:-1]
                if cleaned_doc_lines and not (cleaned_doc_lines[0].strip().startswith('"""') or cleaned_doc_lines[0].strip().startswith('/**')): pass
                generated_doc = "\n".join(cleaned_doc_lines).strip()

        # --- Inline Evaluation & Langfuse Scoring ---
        if EVALUATION_SCRIPT_AVAILABLE and current_trace and generated_doc:
            print(f"Running inline evaluation for trace ID: {current_trace.id}")
            try:
                eval_results = evaluate_documentation(code_snippet, language, generated_doc)
                print(f"Evaluation results: {eval_results}")
                for metric_name, metric_value in eval_results.items():
                    if metric_name in ["Code_Analysis_Debug", "Notes"]: # Skip non-score fields
                        continue
                    
                    score_val_for_langfuse = 0 # Default numeric score
                    comment_for_langfuse = str(metric_value)

                    if isinstance(metric_value, bool):
                        score_val_for_langfuse = 1 if metric_value else 0
                    elif isinstance(metric_value, str):
                        # For categorical strings, we can try to map them or just use a default numeric value
                        # and put the string in the comment.
                        # Example mapping for readability:
                        if metric_name == "M5_Word_Count_Readability":
                            if metric_value == "Acceptable Length": score_val_for_langfuse = 1
                            elif metric_value == "Too Short": score_val_for_langfuse = 0.5 # Or some other numeric mapping
                            elif metric_value == "Too Long": score_val_for_langfuse = 0.5
                            # else "Not Applicable" will be 0 by default
                        elif "Documentation" in metric_name: # For M2 and M3
                            if "Present and Expected" in metric_value: score_val_for_langfuse = 1
                            elif "Absent and Not Expected" in metric_value: score_val_for_langfuse = 1 # Also good
                            elif "Missing but Expected" in metric_value: score_val_for_langfuse = 0.25 # Penalize
                            elif "Present but Not Expected" in metric_value: score_val_for_langfuse = 0.75 # Slightly penalized
                            # else "Not Applicable" will be 0

                    try:
                        print(f"Logging score to Langfuse: TraceID={current_trace.id}, Name='{metric_name}', Value={score_val_for_langfuse}, Comment='{comment_for_langfuse}'")
                        current_trace.score(
                            name=metric_name, # e.g., "M1_Summary_Present"
                            value=score_val_for_langfuse, # Numeric value
                            comment=comment_for_langfuse # Original string value as comment
                        )
                    except Exception as e_score:
                        print(f"Langfuse error logging score '{metric_name}': {e_score}")
            except Exception as e_eval_call:
                print(f"Error calling evaluate_documentation or processing its results: {e_eval_call}")
        # --- End Inline Evaluation ---

        if current_trace: 
            try: current_trace.update(output={"generated_documentation": generated_doc}, level="DEFAULT")
            except Exception as e: print(f"Langfuse error updating trace output (non-critical): {e}")

        return DocumentationOutput(
            message="Documentation generated successfully (from AI).",
            original_code=code_snippet,
            generated_documentation=generated_doc
        )
    except Exception as e:
        # ... (your existing comprehensive error handling for the main try block) ...
        if generation_span: 
            try: generation_span.end(level="ERROR", status_message=str(e))
            except Exception as le: print(f"Langfuse error ending generation span with error (non-critical): {le}")
        if current_trace:  
            try: current_trace.update(level="ERROR", status_message=str(e), output={"error": str(e)})
            except Exception as le: print(f"Langfuse error updating trace with error (non-critical): {le}")
        print(f"Error during API processing: {type(e).__name__} - {str(e)}") 
        if isinstance(e, HTTPException): raise
        else: raise HTTPException(status_code=503, detail="AI service unavailable or encountered an error during generation.")