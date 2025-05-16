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
    # main.py (inside the generate_docs function)

    # ... (input_data: CodeInput) ...
    if not client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized. Check API key.")

    language = input_data.language.lower() if input_data.language else "unknown"
    code_snippet = input_data.code

    # --- START MODIFIED PROMPT SECTION ---
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
    # --- END MODIFIED PROMPT SECTION ---


    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a precise code documentation generator."}, # System role can also be refined
                {"role": "user", "content": prompt}
            ],
            temperature=0.2 # Slightly lower for more structured output
        )
        generated_doc = completion.choices[0].message.content.strip()
        # Optional: Add a simple cleanup if the model sometimes includes the ``` code block markers
        if generated_doc.startswith("```") and generated_doc.endswith("```"):
            lines = generated_doc.split('\n')
            if len(lines) > 2: # Ensure there's content between the markers
                # Remove first line (e.g., ```python) and last line (```)
                cleaned_doc_lines = lines[1:-1]
                # Check if the first line of the supposed doc is actually the docstring/comment start
                # This check is naive and might need refinement
                if not (cleaned_doc_lines[0].strip().startswith('"""') or cleaned_doc_lines[0].strip().startswith('/**')):
                    # If the model just wrapped its output in triple backticks without the language specifier on the first line
                    pass # Keep as is, the strip() might have handled it.
                generated_doc = "\n".join(cleaned_doc_lines).strip()


        return DocumentationOutput(
            message="Documentation generated successfully.",
            original_code=code_snippet,
            generated_documentation=generated_doc
        )
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate documentation from AI: {str(e)}")