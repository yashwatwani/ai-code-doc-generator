import os
from fastapi import FastAPI, HTTPException, Security, Request
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# --- Add CORS Middleware ---
from fastapi.middleware.cors import CORSMiddleware # Import CORSMiddleware

load_dotenv()

# --- Environment Variables ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EXPECTED_API_KEY = os.getenv("MY_APP_API_KEY") 

# --- Rate Limiter Setup ---
limiter = Limiter(key_func=get_remote_address, default_limits=["5/minute"])

# --- OpenAI Client Initialization ---
try:
    if not OPENAI_API_KEY:
        print("Warning: OPENAI_API_KEY environment variable not set. OpenAI client will not be functional.")
        client = None
    else:
        client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    client = None

# --- API Key Authentication Setup ---
API_KEY_NAME = "X-API-Key"
api_key_header_auth = APIKeyHeader(name=API_KEY_NAME, auto_error=False) 

async def get_api_key(api_key_header: str | None = Security(api_key_header_auth)):
    if api_key_header is None:
        raise HTTPException(status_code=403, detail="Not authenticated: X-API-Key header missing.")
    if not EXPECTED_API_KEY: 
        print("Error: Server misconfiguration - MY_APP_API_KEY is not set.")
        raise HTTPException(status_code=500, detail="API Key authentication is not configured correctly on the server.")
    if api_key_header == EXPECTED_API_KEY:
        return api_key_header
    else:
        raise HTTPException(status_code=403, detail="Could not validate credentials")

# --- Pydantic Models ---
class CodeInput(BaseModel):
    code: str
    language: str | None = None

class DocumentationOutput(BaseModel):
    message: str
    original_code: str
    generated_documentation: str | None = None

# --- FastAPI App Instance ---
app = FastAPI()

# --- Add Middlewares (CORS should usually be one of the first) ---
origins = [
    "http://localhost:3000", # Your Next.js frontend development server
    "http://localhost:3001", # If your Next.js runs on 3001
    # Add any other origins you want to allow (e.g., your deployed frontend URL)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "PUT", "DELETE", "PATCH"], # Be more explicit
    allow_headers=["X-API-Key", "Content-Type", "Authorization", "Accept", "Origin", "X-Requested-With"], # Explicitly list common and your custom headers
)

# --- Add Rate Limiter to the App ---
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- API Endpoints ---
# ... (rest of your main.py, including endpoints, remains the same) ...
@app.get("/")
@limiter.limit("20/minute") 
async def read_root(request: Request): 
    return {"message": "Welcome to the AI Code Documentation Generator API"}

@app.post("/generate-documentation/", response_model=DocumentationOutput, dependencies=[Security(get_api_key)])
@limiter.limit("5/minute") 
async def generate_docs(request: Request, input_data: CodeInput): 
    if not client:
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
            "Example for Python:",
            "\"\"\"",
            "Summary of the function.",
            "",
            "Args:",
            "    param_name (param_type): Description of the parameter.",
            "",
            "Returns:",
            "    return_type: Description of the return value.",
            "\"\"\""
        ])
    elif language == "javascript":
        prompt_parts.extend([
            "\nFor JavaScript, format the documentation as a JSDoc comment block.",
            "Example for JavaScript:",
            "/**",
            " * Summary of the function.",
            " *",
            " * @param {param_type} param_name - Description of the parameter.",
            " * @returns {return_type} Description of the return value.",
            " */"
        ])
    else:
        prompt_parts.extend([
            "\nFor this language, use a standard block comment style appropriate for the language.",
            "Focus on clarity and completeness regarding summary, parameters, and return values."
        ])
    
    prompt_parts.append("\nBe precise and do not add any conversational fluff or explanations outside the documentation block itself.")
    prompt = "\n".join(prompt_parts)

    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a precise code documentation generator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        generated_doc = completion.choices[0].message.content.strip()
        
        if generated_doc.startswith("```") and generated_doc.endswith("```"):
            lines = generated_doc.split('\n')
            if len(lines) > 2:
                cleaned_doc_lines = lines[1:-1]
                if cleaned_doc_lines and not (cleaned_doc_lines[0].strip().startswith('"""') or cleaned_doc_lines[0].strip().startswith('/**')):
                    pass
                generated_doc = "\n".join(cleaned_doc_lines).strip()

        return DocumentationOutput(
            message="Documentation generated successfully.",
            original_code=code_snippet,
            generated_documentation=generated_doc
        )
    except Exception as e:
        print(f"Error during OpenAI API call: {type(e).__name__} - {str(e)}")
        raise HTTPException(status_code=503, detail="AI service unavailable or encountered an error during generation.")