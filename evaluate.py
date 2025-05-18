# evaluate.py
import re
# If you want to try Flesch Reading Ease, you might need a library like 'textstat'
# pip install textstat
# import textstat 

def check_presence_of_summary(documentation: str, language: str) -> bool:
    """
    Checks if a general summary seems to be present.
    Very basic: looks for non-empty content before common section headers.
    """
    if not documentation or not documentation.strip():
        return False
    
    doc_lower = documentation.lower()
    # Common section headers
    section_headers = ["args:", "parameters:", "@param", "returns:", "@returns"]
    
    first_section_index = len(documentation) # Default to end of string
    for header in section_headers:
        try:
            idx = doc_lower.index(header)
            if idx < first_section_index:
                first_section_index = idx
        except ValueError:
            continue # Header not found
            
    summary_part = documentation[:first_section_index].strip()
    # A summary should have a reasonable length, e.g., more than a few words
    return len(summary_part.split()) > 3 

def check_presence_of_parameters_section(documentation: str, language: str) -> bool:
    """Checks for 'Args:', 'Parameters:', or '@param'."""
    doc_lower = documentation.lower()
    if language == "python":
        return "args:" in doc_lower
    elif language == "javascript":
        return "@param" in doc_lower
    # Add more language checks or a general check if needed
    return "parameters:" in doc_lower or "args:" in doc_lower or "@param" in doc_lower

def check_presence_of_returns_section(documentation: str, language: str) -> bool:
    """Checks for 'Returns:' or '@returns'."""
    doc_lower = documentation.lower()
    if language == "python":
        return "returns:" in doc_lower
    elif language == "javascript":
        return "@returns" in doc_lower
    # Add more language checks or a general check
    return "returns:" in doc_lower or "@returns" in doc_lower

def check_basic_format_adherence(documentation: str, language: str) -> bool:
    """Checks for basic start/end markers."""
    doc_stripped = documentation.strip()
    if language == "python":
        return doc_stripped.startswith('"""') and doc_stripped.endswith('"""')
    elif language == "javascript":
        return doc_stripped.startswith('/**') and doc_stripped.endswith('*/')
    # For other languages, this might be harder to define simply
    return True # Default to True if no specific check

def check_word_count_readability(documentation: str, min_words=10, max_words=150) -> str:
    """Categorical readability based on word count."""
    word_count = len(documentation.strip().split())
    if word_count < min_words:
        return "Too Short"
    elif word_count > max_words:
        return "Too Long"
    return "Acceptable Length"

# Example function to analyze code structure (very basic)
def analyze_code_structure(code: str, language: str) -> dict:
    """
    Very basic analysis to guess if params/returns are expected.
    This is a placeholder and would need significant improvement for real use.
    """
    has_params = False
    has_return = False
    
    if language == "python":
        if "def " in code and "(" in code and ")" in code and \
           code[code.find("(") + 1:code.find(")")].strip() != "":
            has_params = True # Rough guess
        if "return " in code:
            has_return = True
    elif language == "javascript":
        if "function " in code and "(" in code and ")" in code and \
           code[code.find("(") + 1:code.find(")")].strip() != "":
            has_params = True # Rough guess
        if "return " in code:
            has_return = True
            
    return {"expected_params": has_params, "expected_return": has_return}


# evaluate.py

# ... (keep all check_... helper functions as they are) ...
# ... (keep analyze_code_structure as it is, acknowledging its basic nature) ...

def evaluate_documentation(code: str, language: str, generated_doc: str) -> dict:
    """
    Evaluates generated documentation based on defined metrics.
    """
    if not generated_doc or not generated_doc.strip():
        return {
            "M1_Summary_Present": False,
            "M2_Parameter_Documentation": "Not Applicable (No Doc)", # Changed
            "M3_Return_Documentation": "Not Applicable (No Doc)",    # Changed
            "M4_Basic_Format_Adherence": False,
            "M5_Word_Count_Readability": "Not Applicable (No Doc)",
            "Notes": "No documentation generated."
        }

    language = language.lower() if language else "unknown"
    code_analysis = analyze_code_structure(code, language) # expected_params, expected_return

    m1_summary = check_presence_of_summary(generated_doc, language)
    
    # Metric 2: Parameter Documentation
    params_section_actually_present = check_presence_of_parameters_section(generated_doc, language)
    if code_analysis["expected_params"]:
        if params_section_actually_present:
            m2_param_doc = "Present and Expected"
        else:
            m2_param_doc = "Missing but Expected"
    else: # Not expecting params
        if params_section_actually_present:
            m2_param_doc = "Present but Not Expected (e.g., 'Args: None')"
        else:
            m2_param_doc = "Absent and Not Expected"
            
    # Metric 3: Return Documentation
    returns_section_actually_present = check_presence_of_returns_section(generated_doc, language)
    if code_analysis["expected_return"]:
        if returns_section_actually_present:
            m3_return_doc = "Present and Expected"
        else:
            m3_return_doc = "Missing but Expected"
    else: # Not expecting return
        if returns_section_actually_present:
            m3_return_doc = "Present but Not Expected (e.g., 'Returns: None')"
        else:
            m3_return_doc = "Absent and Not Expected"

    m4_format = check_basic_format_adherence(generated_doc, language)
    m5_readability = check_word_count_readability(generated_doc)
    
    return {
        "M1_Summary_Present": m1_summary,
        "M2_Parameter_Documentation": m2_param_doc,
        "M3_Return_Documentation": m3_return_doc,
        "M4_Basic_Format_Adherence": m4_format,
        "M5_Word_Count_Readability": m5_readability,
        "Code_Analysis_Debug": code_analysis 
    }

# ... (keep the if __name__ == "__main__": block for testing) ...

if __name__ == "__main__":
    # --- Example Test Cases ---
    print("--- Test Case 1: Python Function with Args and Return ---")
    code1 = """
def add(a: int, b: int) -> int:
    return a + b
"""
    doc1 = """\"\"\"
Adds two numbers.

Args:
    a (int): The first number.
    b (int): The second number.

Returns:
    int: The sum of a and b.
\"\"\""""
    eval1 = evaluate_documentation(code1, "python", doc1)
    print(eval1)

    print("\n--- Test Case 2: JavaScript Function, No Explicit Return in Doc ---")
    code2 = """
function greet(name) {
    console.log("Hello, " + name);
}
"""
    doc2 = """/**
 * Greets a person.
 * @param {string} name - The name of the person.
 */"""
    eval2 = evaluate_documentation(code2, "javascript", doc2)
    print(eval2)

    print("\n--- Test Case 3: Python Function, Simple, No Args/Return ---")
    code3 = """
def say_hello():
    print("Hello world")
"""
    # LLM might still generate Args:None, Returns:None based on our prompt
    doc3 = """\"\"\"
Prints 'Hello world'.

Args:
    None

Returns:
    None
\"\"\""""
    eval3 = evaluate_documentation(code3, "python", doc3)
    print(eval3)

    print("\n--- Test Case 4: Bad Formatting (Python) ---")
    doc4 = """
    This is a summary.
    Args: nope
    Returns: maybe
    """
    eval4 = evaluate_documentation(code1, "python", doc4) # Using code1 for structure
    print(eval4)
    
    print("\n--- Test Case 5: Too Short ---")
    doc5 = "\"\"\"Hi.\"\"\""
    eval5 = evaluate_documentation(code1, "python", doc5)
    print(eval5)
    
    print("\n--- Test Case 6: Empty Doc ---")
    doc6 = ""
    eval6 = evaluate_documentation(code1, "python", doc6)
    print(eval6)