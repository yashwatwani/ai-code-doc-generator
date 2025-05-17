import os
from dotenv import load_dotenv

# Load .env as early as possible
load_dotenv() 

# --- Langfuse Integration ---
from langfuse import Langfuse

LANGFUSE_PUBLIC_KEY_VAR = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY_VAR = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST_VAR = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

langfuse_client_for_tracing = None 
try:
    if LANGFUSE_PUBLIC_KEY_VAR and LANGFUSE_SECRET_KEY_VAR:
        langfuse_client_for_tracing = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY_VAR,
            secret_key=LANGFUSE_SECRET_KEY_VAR,
            host=LANGFUSE_HOST_VAR
            # debug=False # Set to False or remove for production
        )
        print("Langfuse client initialized.") # Keep this for startup confirmation
    else:
        print("Warning: Langfuse environment variables (PUBLIC_KEY, SECRET_KEY) not fully set. Langfuse tracing will be disabled.")
except Exception as e:
    print(f"Error initializing Langfuse client: {type(e).__name__} - {e}")

# --- OpenAI Client Initialization ---
from openai import OpenAI 

OPENAI_API_KEY_VAR = os.getenv("OPENAI_API_KEY")
openai_llm_client = None 
try:
    if not OPENAI_API_KEY_VAR:
        print("Warning: OPENAI_API_KEY environment variable not set. OpenAI client will not be functional.")
    else:
        openai_llm_client = OpenAI(api_key=OPENAI_API_KEY_VAR)
        print("OpenAI client initialized.") # Keep this for startup confirmation
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
EXPECTED_API_KEY_VAR = os.getenv("MY_APP_API_KEY")

# Rate Limiter
limiter = Limiter(key_func=get_remote_address, default_limits=["5/minute"])

# API Key Authentication Setup
API_KEY_NAME = "X-API-Key"
api_key_header_auth = APIKeyHeader(name=API_KEY_NAME, auto_error=False) 

async def get_api_key(api_key_header: str | None = Security(api_key_header_auth)):
    if api_key_header is None:
        raise HTTPException(status_code=403, detail="Not authenticated: X-API-Key header missing.")
    if not EXPECTED_API_KEY_VAR: 
        # Server-side log for this specific misconfiguration
        print("CRITICAL SERVER ERROR: MY_APP_API_KEY (for client authentication) is not set in the environment.")
        raise HTTPException(status_code=500, detail="API Key authentication is not configured correctly on the server.")
    if api_key_header == EXPECTED_API_KEY_VAR:
        return api_key_header
    else:
        raise HTTPException(status_code=403, detail="Could not validate credentials")

# Pydantic Models
class CodeInput(BaseModel):
    code: str
    language: str | None = None

class DocumentationOutput(BaseModel):
    message: str
    original_code: str
    generated_documentation: str | None = None

# FastAPI App Instance
app = FastAPI()

# Middlewares
origins = [ 
    "http://localhost:3000", 
    "http://localhost:3001",
]
app.add_middleware(
    CORSMiddleware, 
    allow_origins=origins, 
    allow_credentials=True, 
    allow_methods=["GET", "POST", "OPTIONS"], 
    allow_headers=["X-API-Key", "Content-Type", "Authorization"]
)
                   
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# --- API Endpoints ---
@app.get("/")
@limiter.limit("20/minute") 
async def read_root(request: Request): 
    if langfuse_client_for_tracing:
        try:
            langfuse_client_for_tracing.trace(
                name="read_root_trace", 
                user_id=request.client.host if request.client else "unknown_client"
            )
        except Exception as e:
            print(f"Langfuse error in read_root (non-critical): {e}") # Log non-critical Langfuse errors
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
        except Exception as e:
            print(f"Langfuse error starting trace in generate_docs (non-critical): {e}")

    if not openai_llm_client: 
        if current_trace: current_trace.update(level="ERROR", status_message="OpenAI client not initialized", output={"error": "OpenAI client misconfiguration"})
        raise HTTPException(status_code=503, detail="AI service client not initialized. Check server configuration for OpenAI API key.")

    language = input_data.language.lower() if input_data.language else "unknown"
    code_snippet = input_data.code
    
    prompt_parts = [
        "You are an expert programmer tasked with generating high-quality, structured documentation for code.",
        f"The language of the code is: {language}.",
        "Analyze the following code snippet:",
        f"```\n{code_snippet}\n```",
        "\nGenerate documentation that includes:",
        "1. A concise summary of what the function/code does.",
        "2. A description of its parameters (name, type, description), if any.",
        "3. A description of what it returns (type, description), if any."
    ]
    if language == "python":
        prompt_parts.extend([
            "\nFor Python, format the documentation as a standard PEP 257 multiline docstring.",
            "Example for Python:", "\"\"\"", "Summary of the function.", "", "Args:",
            "    param_name (param_type): Description of the parameter.", "", "Returns:",
            "    return_type: Description of the return value.", "\"\"\""
        ])
    elif language == "javascript":
        prompt_parts.extend([
            "\nFor JavaScript, format the documentation as a JSDoc comment block.",
            "Example for JavaScript:", "/**", " * Summary of the function.", " *",
            " * @param {param_type} param_name - Description of the parameter.",
            " * @returns {return_type} Description of the return value.", " */"
        ])
    else:
        prompt_parts.extend([
            "\nFor this language, use a standard block comment style appropriate for the language.",
            "Focus on clarity and completeness regarding summary, parameters, and return values."
        ])
    prompt_parts.append("\nBe precise and do not add any conversational fluff or explanations outside the documentation block itself.")
    prompt = "\n".join(prompt_parts)

    if current_trace:
        try:
            generation_span = current_trace.generation(
                name="openai-documentation-generation",
                model="gpt-3.5-turbo",
                model_parameters={"temperature": 0.2},
                prompt=prompt 
            )
        except Exception as e:
            print(f"Langfuse error starting generation span (non-critical): {e}")
            generation_span = None

    try:
        completion_obj = openai_llm_client.chat.completions.create( 
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a precise code documentation generator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        generated_doc = completion_obj.choices[0].message.content.strip()
        
        openai_usage_data = None 
        if hasattr(completion_obj, 'usage') and completion_obj.usage is not None:
            openai_usage_data = completion_obj.usage 
        
        if generation_span:
            try:
                generation_span.end(
                    output=generated_doc,
                    usage=openai_usage_data 
                )
            except Exception as e:
                print(f"Langfuse error ending generation span (non-critical): {e}")
        
        # Backtick cleanup
        if generated_doc.startswith("```") and generated_doc.endswith("```"):
            lines = generated_doc.split('\n')
            if len(lines) > 2:
                cleaned_doc_lines = lines[1:-1]
                if cleaned_doc_lines and not (cleaned_doc_lines[0].strip().startswith('"""') or cleaned_doc_lines[0].strip().startswith('/**')):
                    pass
                generated_doc = "\n".join(cleaned_doc_lines).strip()

        if current_trace: 
            try:
                current_trace.update(output={"generated_documentation": generated_doc}, level="DEFAULT")
            except Exception as e:
                print(f"Langfuse error updating trace output (non-critical): {e}")


        return DocumentationOutput(
            message="Documentation generated successfully.",
            original_code=code_snippet,
            generated_documentation=generated_doc
        )
    except Exception as e:
        if generation_span: # If generation_span was created, try to end it with error
            try: generation_span.end(level="ERROR", status_message=str(e))
            except Exception as le: print(f"Langfuse error ending generation span with error (non-critical): {le}")
        if current_trace:  # Also update the main trace with error
            try:
                current_trace.update(level="ERROR", status_message=str(e), output={"error": str(e)})
            except Exception as le:
                print(f"Langfuse error updating trace with error (non-critical): {le}")
        
        # Log the original error for server-side debugging
        print(f"Error during API processing: {type(e).__name__} - {str(e)}") 
        if isinstance(e, HTTPException): # If it's already an HTTPException, re-raise
            raise
        else: # For other exceptions, wrap in a generic 503 for the client
            raise HTTPException(status_code=503, detail="AI service unavailable or encountered an error during generation.")