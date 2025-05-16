import os # For accessing environment variables
from fastapi import FastAPI, HTTPException # Import HTTPException
from pydantic import BaseModel
from openai import OpenAI # Import the OpenAI client
from dotenv import load_dotenv # Import dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI client
# It will automatically look for the OPENAI_API_KEY environment variable
try:
    client = OpenAI()
except Exception as e:
    # Handle cases where API key might not be set or other initialization errors
    print(f"Error initializing OpenAI client: {e}")
    client = None


# Define a Pydantic model for the request body
class CodeInput(BaseModel):
    code: str
    language: str | None = None

# Define a Pydantic model for the response body
class DocumentationOutput(BaseModel):
    message: str
    original_code: str
    generated_documentation: str | None = None

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Welcome to the AI Code Documentation Generator API"}

@app.post("/generate-documentation/", response_model=DocumentationOutput)
async def generate_docs(input_data: CodeInput):
    if not client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized. Check API key.")

    language_context = f"for the following {input_data.language or 'code'} snippet" if input_data.language else "for the following code snippet"

    prompt = f"""
    You are an expert programmer tasked with generating clear and concise documentation for code.
    Generate a brief documentation string or comment block {language_context}:

    ```
    {input_data.code}
    ```

    The documentation should explain what the code does, its parameters (if any), and what it returns (if any).
    If the language is Python, generate a standard Python docstring. For other languages, use appropriate comment styles.
    Be brief and focus on the core functionality.
    """

    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo", # Or another model like "gpt-4" if you have access
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates code documentation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3 # Lower temperature for more deterministic output
        )
        generated_doc = completion.choices[0].message.content.strip()

        return DocumentationOutput(
            message="Documentation generated successfully.",
            original_code=input_data.code,
            generated_documentation=generated_doc
        )
    except Exception as e:
        print(f"Error calling OpenAI API: {e}") # Log the error for debugging
        raise HTTPException(status_code=500, detail=f"Failed to generate documentation from AI: {str(e)}")