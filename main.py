import os
from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# --- Environment Variables ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EXPECTED_API_KEY = os.getenv("MY_APP_API_KEY") # Used by get_api_key dependency

# --- OpenAI Client Initialization ---
try:
    if not OPENAI_API_KEY:
        # This check is for app startup. If key is missing, client won't be functional.
        print("Warning: OPENAI_API_KEY environment variable not set. OpenAI client will not be functional.")
        client = None
    else:
        client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    client = None


# --- API Key Authentication Setup ---
API_KEY_NAME = "X-API-Key"
# Set auto_error to False to handle missing header explicitly in get_api_key
api_key_header_auth = APIKeyHeader(name=API_KEY_NAME, auto_error=False) 

async def get_api_key(api_key_header: str | None = Security(api_key_header_auth)):
    """
    Dependency that checks for a valid API key in the X-API-Key header.
    """
    if api_key_header is None: # Manually check if header was provided
        raise HTTPException(status_code=403, detail="Not authenticated: X-API-Key header missing.")

    # This EXPECTED_API_KEY is the module-level global from above
    if not EXPECTED_API_KEY: 
        # This means the server itself is misconfigured (MY_APP_API_KEY not in .env)
        raise HTTPException(status_code=500, detail="API Key not configured on server.")
    
    if api_key_header == EXPECTED_API_KEY:
        return api_key_header
    else:
        raise HTTPException(
            status_code=403, detail="Could not validate credentials"
        )

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

# --- API Endpoints ---
@app.get("/")
async def read_root():
    return {"message": "Welcome to the AI Code Documentation Generator API"}

@app.post("/generate-documentation/", response_model=DocumentationOutput, dependencies=[Security(get_api_key)])
async def generate_docs(input_data: CodeInput):
    if not client:
        # This means OpenAI client couldn't initialize (e.g., missing OPENAI_API_KEY or invalid key format)
        raise HTTPException(status_code=500, detail="OpenAI client not initialized. Check server configuration for OpenAI API key.")

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
                if not (cleaned_doc_lines[0].strip().startswith('"""') or cleaned_doc_lines[0].strip().startswith('/**')):
                    pass
                generated_doc = "\n".join(cleaned_doc_lines).strip()

        return DocumentationOutput(
            message="Documentation generated successfully.",
            original_code=code_snippet,
            generated_documentation=generated_doc
        )
    except Exception as e:
        # Log the actual error for server-side debugging
        print(f"Error during OpenAI API call: {type(e).__name__} - {str(e)}")
        # Return a generic error to the client
        raise HTTPException(status_code=503, detail="AI service unavailable or encountered an error.")