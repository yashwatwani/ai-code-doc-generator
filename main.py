import os
from dotenv import load_dotenv
load_dotenv() 

# --- Cachetools for In-Memory Caching ---
from cachetools import TTLCache # Import TTLCache

# ... (other imports: Langfuse, OpenAI, FastAPI, etc. remain the same) ...
from langfuse import Langfuse
from openai import OpenAI 
from fastapi import FastAPI, HTTPException, Security, Request
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.middleware.cors import CORSMiddleware

# --- Environment Variables & Client Initializations (as before) ---
# Langfuse
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
langfuse_client_for_tracing = None 
try: # ... (Langfuse init logic as before) ...
    if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
        langfuse_client_for_tracing = Langfuse(public_key=LANGFUSE_PUBLIC_KEY, secret_key=LANGFUSE_SECRET_KEY, host=LANGFUSE_HOST)
        print("Langfuse client initialized.")
    else: print("Warning: Langfuse environment variables not fully set. Langfuse tracing will be disabled.")
except Exception as e: print(f"Error initializing Langfuse client: {type(e).__name__} - {e}")

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_llm_client = None 
try: # ... (OpenAI init logic as before) ...
    if not OPENAI_API_KEY: print("Warning: OPENAI_API_KEY environment variable not set.")
    else:
        openai_llm_client = OpenAI(api_key=OPENAI_API_KEY)
        print("OpenAI client initialized.")
except Exception as e: print(f"Error initializing OpenAI client: {type(e).__name__} - {e}")

# App Specific
EXPECTED_API_KEY = os.getenv("MY_APP_API_KEY") 

# --- Caching Setup ---
# TTLCache: Time To Live Cache. Items expire after a set duration.
# maxsize: Max number of items in cache.
# ttl: Time to live in seconds (e.g., 300 seconds = 5 minutes)
# You can adjust maxsize and ttl based on your needs.
documentation_cache = TTLCache(maxsize=100, ttl=300) 

# ... (Rate Limiter, API Key Auth, Pydantic Models, FastAPI App instance, Middlewares - remain the same) ...
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
origins = [ "http://localhost:3000", "http://localhost:3001", "https://ai-code-doc-generator.vercel.app" ] # Ensure your Vercel URL is here
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["X-API-Key", "Content-Type", "Authorization"])          
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- API Endpoints ---
@app.get("/")
@limiter.limit("20/minute") 
async def read_root(request: Request): 
    # ... (Langfuse trace logic for root if any) ...
    if langfuse_client_for_tracing:
        try: langfuse_client_for_tracing.trace(name="read_root_trace", user_id=request.client.host if request.client else "unknown_client")
        except Exception as e: print(f"Langfuse error in read_root (non-critical): {e}")
    return {"message": "Welcome to the AI Code Documentation Generator API"}

@app.post("/generate-documentation/", response_model=DocumentationOutput, dependencies=[Security(get_api_key)])
@limiter.limit("5/minute") 
async def generate_docs(request: Request, input_data: CodeInput):
    current_trace = None
    generation_span = None # For the OpenAI call
    cache_hit = False # To track if we used cache

    language = input_data.language.lower() if input_data.language else "unknown"
    code_snippet = input_data.code

    # --- Caching Logic: Check cache first ---
    cache_key = (code_snippet, language) # Use a tuple of code and language as the cache key
    if cache_key in documentation_cache:
        cached_documentation = documentation_cache[cache_key]
        print(f"Cache HIT for key: ({language}, code_hash_placeholder)") # Avoid logging full code
        cache_hit = True
        
        # Create a Langfuse trace even for cache hits for observability
        if langfuse_client_for_tracing:
            try:
                current_trace = langfuse_client_for_tracing.trace(
                    name="generate-code-documentation-cache-hit", # Differentiate cache hit traces
                    user_id=request.client.host if request.client else "unknown_client",
                    metadata={"language": language, "code_length": len(code_snippet), "cache_hit": True},
                    tags=["core-feature", "cache-hit", f"lang:{language}"]
                )
                current_trace.update(
                    input={"code": code_snippet, "language": language},
                    output={"generated_documentation": cached_documentation}
                )
            except Exception as e: print(f"Langfuse error during cache hit trace: {e}")

        return DocumentationOutput(
            message="Documentation retrieved from cache successfully.",
            original_code=code_snippet,
            generated_documentation=cached_documentation
        )
    # --- End Caching Logic: Check cache ---
    
    print(f"Cache MISS for key: ({language}, code_hash_placeholder)") # Avoid logging full code

    # If not a cache hit, proceed with Langfuse trace creation and OpenAI call
    if langfuse_client_for_tracing:
        try:
            current_trace = langfuse_client_for_tracing.trace(
                name="generate-code-documentation", # Original trace name for cache misses
                user_id=request.client.host if request.client else "unknown_client",
                metadata={"language": language, "code_length": len(code_snippet), "cache_hit": False},
                tags=["core-feature", "cache-miss", f"lang:{language}"]
            )
            current_trace.update(input={"code": code_snippet, "language": language})
        except Exception as e: print(f"Langfuse error starting trace: {e}")

    if not openai_llm_client: 
        if current_trace: current_trace.update(level="ERROR", status_message="OpenAI client not initialized", output={"error": "OpenAI client misconfiguration"})
        raise HTTPException(status_code=503, detail="AI service client not initialized. Check server configuration for OpenAI API key.")

    prompt_parts = [ # ... (your full prompt_parts logic as before) ... 
        "You are an expert programmer...", f"The language of the code is: {language}.", f"```\n{code_snippet}\n```",
        "\nGenerate documentation that includes:", "1. A concise summary...", "2. A description of its parameters...", "3. A description of what it returns..."
    ]
    if language == "python": prompt_parts.extend(["\nFor Python, format...", "Example for Python:...", "\"\"\"...", "\"\"\""]) 
    elif language == "javascript": prompt_parts.extend(["\nFor JavaScript, format...", "Example for JavaScript:...", "/**...", " */"]) 
    else: prompt_parts.extend(["\nFor this language, use a standard block comment..."]) 
    prompt_parts.append("\nBe precise and do not add any conversational fluff...")
    prompt = "\n".join(prompt_parts)

    if current_trace:
        try:
            generation_span = current_trace.generation(
                name="openai-documentation-generation", model="gpt-3.5-turbo",
                model_parameters={"temperature": 0.2}, prompt=prompt 
            )
        except Exception as e: print(f"Langfuse error starting generation span: {e}"); generation_span = None

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
            except Exception as e: print(f"Langfuse error ending generation span: {e}")
        
        # --- Caching Logic: Store result in cache ---
        documentation_cache[cache_key] = generated_doc
        print(f"Stored in cache for key: ({language}, code_hash_placeholder)")
        # --- End Caching Logic: Store result ---

        # ... (backtick cleanup as before) ...
        if generated_doc.startswith("```") and generated_doc.endswith("```"):
            lines = generated_doc.split('\n'); # ... (rest of cleanup) ...
            if len(lines) > 2:
                cleaned_doc_lines = lines[1:-1]
                if cleaned_doc_lines and not (cleaned_doc_lines[0].strip().startswith('"""') or cleaned_doc_lines[0].strip().startswith('/**')): pass
                generated_doc = "\n".join(cleaned_doc_lines).strip()

        if current_trace: 
            try: current_trace.update(output={"generated_documentation": generated_doc}, level="DEFAULT")
            except Exception as e: print(f"Langfuse error updating trace output: {e}")

        return DocumentationOutput(
            message="Documentation generated successfully (from AI).", # Indicate source
            original_code=code_snippet,
            generated_documentation=generated_doc
        )
    except Exception as e:
        # ... (error handling as before, including updating Langfuse trace/generation on error) ...
        if generation_span: 
            try: generation_span.end(level="ERROR", status_message=str(e))
            except Exception as le: print(f"Langfuse error ending generation span with error: {le}")
        if current_trace:  
            try: current_trace.update(level="ERROR", status_message=str(e), output={"error": str(e)})
            except Exception as le: print(f"Langfuse error updating trace with error: {le}")
        print(f"Error during API processing: {type(e).__name__} - {str(e)}") 
        if isinstance(e, HTTPException): raise
        else: raise HTTPException(status_code=503, detail="AI service unavailable or encountered an error during generation.")