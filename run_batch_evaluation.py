import json
import requests # To call your deployed API
import time
import re # Needed for some evaluation functions, ensure it's imported

# --- Configuration for calling your deployed API ---
# REPLACE with your actual Render URL and the API key your Render backend expects
BACKEND_API_URL = "https://ai-code-doc-generator.onrender.com/generate-documentation/" 
BACKEND_API_KEY = "your_secret_app_api_key_123" # !!! REPLACE THIS !!!

# --- Evaluation Metric Helper Functions & Main Evaluator ---
# (Copied from your previous evaluate.py - ensure these are up-to-date with your latest logic)

def analyze_code_structure(code: str, language: str) -> dict:
    """
    Very basic analysis to guess if params/returns are expected.
    This is a placeholder and would need significant improvement for real use.
    """
    has_params = False
    has_return = False
    
    # Ensure language is lowercased for consistent matching
    lang_lower = language.lower() if language else "unknown"

    if lang_lower == "python":
        if "def " in code and "(" in code and ")" in code:
            params_str = code[code.find("(") + 1 : code.find(")")]
            if params_str.strip() != "" and params_str.strip().lower() != "self": # Check if only self
                 # Further check if 'self' is the only thing after stripping, if so, not user-params
                if not (params_str.strip().lower() == "self" and "," not in params_str):
                    has_params = True 
        if "return " in code:
            has_return = True
    elif lang_lower == "javascript":
        if "function " in code and "(" in code and ")" in code:
            params_str = code[code.find("(") + 1 : code.find(")")]
            if params_str.strip() != "":
                has_params = True
        if "return " in code:
            has_return = True
            
    return {"expected_params": has_params, "expected_return": has_return}

def check_presence_of_summary(documentation: str, language: str) -> bool:
    if not documentation or not documentation.strip():
        return False
    doc_lower = documentation.lower()
    section_headers = ["args:", "parameters:", "@param", "returns:", "@returns", "params:"]
    first_section_index = len(documentation)
    for header in section_headers:
        try:
            idx = doc_lower.index(header)
            if idx < first_section_index:
                first_section_index = idx
        except ValueError:
            continue
    summary_part = documentation[:first_section_index].strip()
    # Consider a summary valid if it's not just the start/end markers of the docstring
    if language == "python" and (summary_part == '"""' or summary_part == "'''"): return False
    if language == "javascript" and summary_part == '/**': return False
    return len(summary_part.split()) > 3

def check_presence_of_parameters_section(documentation: str, language: str) -> bool:
    doc_lower = documentation.lower()
    if language == "python":
        return "args:" in doc_lower
    elif language == "javascript":
        return "@param" in doc_lower
    return "parameters:" in doc_lower or "args:" in doc_lower or "@param" in doc_lower

def check_presence_of_returns_section(documentation: str, language: str) -> bool:
    doc_lower = documentation.lower()
    if language == "python":
        return "returns:" in doc_lower
    elif language == "javascript":
        return "@returns" in doc_lower
    return "returns:" in doc_lower or "@returns" in doc_lower

def check_basic_format_adherence(documentation: str, language: str) -> bool:
    doc_stripped = documentation.strip()
    if language == "python":
        return doc_stripped.startswith('"""') and doc_stripped.endswith('"""') or \
               doc_stripped.startswith("'''") and doc_stripped.endswith("'''")
    elif language == "javascript":
        return doc_stripped.startswith('/**') and doc_stripped.endswith('*/')
    return True 

def check_word_count_readability(documentation: str, min_words=10, max_words=200) -> str: # Increased max_words
    # Remove common docstring/comment markers before counting words for readability
    cleaned_doc = documentation.strip()
    if cleaned_doc.startswith('"""') and cleaned_doc.endswith('"""'):
        cleaned_doc = cleaned_doc[3:-3].strip()
    elif cleaned_doc.startswith("'''") and cleaned_doc.endswith("'''"):
        cleaned_doc = cleaned_doc[3:-3].strip()
    elif cleaned_doc.startswith('/**') and cleaned_doc.endswith('*/'):
        cleaned_doc = cleaned_doc[3:-2].strip() # Remove /** and */
        # Remove leading * from multiline JS comments
        cleaned_doc_lines = [line.strip().lstrip('*').strip() for line in cleaned_doc.split('\n')]
        cleaned_doc = "\n".join(cleaned_doc_lines).strip()

    if not cleaned_doc: # If only markers were present
        return "Too Short (Essentially Empty)"

    word_count = len(cleaned_doc.split())
    if word_count < min_words:
        return "Too Short"
    elif word_count > max_words:
        return "Too Long"
    return "Acceptable Length"

def evaluate_documentation(code: str, language: str, generated_doc: str) -> dict:
    if not generated_doc or not generated_doc.strip():
        return {
            "M1_Summary_Present": False, "M2_Parameter_Documentation": "Not Applicable (No Doc)",
            "M3_Return_Documentation": "Not Applicable (No Doc)", "M4_Basic_Format_Adherence": False,
            "M5_Word_Count_Readability": "Not Applicable (No Doc)", "Notes": "No documentation generated."
        }

    lang_lower = language.lower() if language else "unknown"
    code_analysis = analyze_code_structure(code, lang_lower)

    m1_summary = check_presence_of_summary(generated_doc, lang_lower)
    
    params_section_actually_present = check_presence_of_parameters_section(generated_doc, lang_lower)
    if code_analysis["expected_params"]:
        m2_param_doc = "Present and Expected" if params_section_actually_present else "Missing but Expected"
    else:
        m2_param_doc = "Present but Not Expected (e.g., 'Args: None')" if params_section_actually_present else "Absent and Not Expected"
            
    returns_section_actually_present = check_presence_of_returns_section(generated_doc, lang_lower)
    if code_analysis["expected_return"]:
        m3_return_doc = "Present and Expected" if returns_section_actually_present else "Missing but Expected"
    else:
        m3_return_doc = "Present but Not Expected (e.g., 'Returns: None')" if returns_section_actually_present else "Absent and Not Expected"

    m4_format = check_basic_format_adherence(generated_doc, lang_lower)
    m5_readability = check_word_count_readability(generated_doc)
    
    return {
        "M1_Summary_Present": m1_summary, "M2_Parameter_Documentation": m2_param_doc,
        "M3_Return_Documentation": m3_return_doc, "M4_Basic_Format_Adherence": m4_format,
        "M5_Word_Count_Readability": m5_readability, "Code_Analysis_Debug": code_analysis 
    }
# --- End Evaluation Functions ---


def call_generation_api(code: str, language: str) -> str | None:
    """Calls your deployed API to get generated documentation."""
    headers = {
        "X-API-Key": BACKEND_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "code": code,
        "language": language
    }
    try:
        # Use .replace('\n', ' ') for cleaner single-line logging of multiline code
        print(f"  Calling API for: {language} snippet (first 70 chars): {code[:70].replace('\n', '↵ ')}...")
        response = requests.post(BACKEND_API_URL, headers=headers, json=payload, timeout=60) 
        response.raise_for_status() 
        data = response.json()
        print("  API call successful.")
        return data.get("generated_documentation")
    except requests.exceptions.Timeout:
        print(f"  Error calling API: Request timed out after 60 seconds.")
        return None
    except requests.exceptions.HTTPError as http_err:
        print(f"  Error calling API: HTTP error occurred: {http_err} - Response: {response.text[:200]}")
        return None
    except requests.exceptions.RequestException as req_err:
        print(f"  Error calling API: {req_err}")
        return None
    except json.JSONDecodeError:
        print(f"  Error decoding JSON response from API. Status: {response.status_code}, Response text: {response.text[:200]}")
        return None


def run_batch_evaluation(dataset_filepath="knowledge_base_data.json"):
    try:
        with open(dataset_filepath, 'r') as f:
            evaluation_dataset = json.load(f)
        print(f"Loaded {len(evaluation_dataset)} items from {dataset_filepath}")
    except Exception as e:
        print(f"Error loading evaluation dataset '{dataset_filepath}': {e}")
        return

    all_eval_results = []
    successful_generations = 0

    for i, item in enumerate(evaluation_dataset):
        item_id = item.get('id', f'item_{i+1}') # Use index if id is missing
        item_lang = item.get('language')
        item_code = item.get('code_snippet')

        if not item_code or not item_lang:
            print(f"\n--- Skipping Item {i+1}/{len(evaluation_dataset)}: {item_id} - Missing 'code' or 'language' key ---")
            all_eval_results.append({"id": item_id, "error": "Missing 'code' or 'language' in dataset item"})
            continue

        print(f"\n--- Evaluating Item {i+1}/{len(evaluation_dataset)}: {item_id} ({item_lang}) ---")
        
        generated_doc = call_generation_api(item_code, item_lang)
        
        if generated_doc is not None: # Check for None explicitly
            successful_generations += 1
            eval_metrics = evaluate_documentation(item_code, item_lang, generated_doc)
            print(f"  Metrics: {eval_metrics}")
            all_eval_results.append({
                "id": item_id, "language": item_lang,
                "code_snippet_preview": item_code[:100] + "...",
                "generated_doc_preview": generated_doc[:100] + "...",
                **eval_metrics
            })
        else:
            print("  Failed to generate documentation from API for this item or documentation was empty.")
            all_eval_results.append({
                "id": item_id, "language": item_lang,
                "error": "Failed to generate via API or received empty documentation"
            })
        
        if i < len(evaluation_dataset) - 1: # Avoid sleep after the last item
             print("  Pausing briefly (12s) to respect API rate limits (5/minute)...")
             time.sleep(12) # 60s / 5 req = 12s per req on average

    print("\n\n--- Overall Batch Evaluation Summary ---")
    num_total_items = len(evaluation_dataset)
    print(f"Total items in dataset: {num_total_items}")
    print(f"Successfully generated documentation for: {successful_generations}/{num_total_items}")

    if successful_generations > 0:
        valid_results = [res for res in all_eval_results if "error" not in res and res.get("M1_Summary_Present") is not None] # Ensure eval ran
        
        summary_present_count = sum(1 for res in valid_results if res.get("M1_Summary_Present") is True)
        if valid_results: print(f"M1_Summary_Present: {summary_present_count}/{len(valid_results)} ({summary_present_count/len(valid_results):.2%})")
        
        format_adherence_count = sum(1 for res in valid_results if res.get("M4_Basic_Format_Adherence") is True)
        if valid_results: print(f"M4_Basic_Format_Adherence: {format_adherence_count}/{len(valid_results)} ({format_adherence_count/len(valid_results):.2%})")

        param_correct_count = sum(1 for res in valid_results if res.get("M2_Parameter_Documentation") in ["Present and Expected", "Absent and Not Expected"])
        if valid_results: print(f"M2_Parameter_Documentation (Correctness): {param_correct_count}/{len(valid_results)} ({param_correct_count/len(valid_results):.2%})")
        
        return_correct_count = sum(1 for res in valid_results if res.get("M3_Return_Documentation") in ["Present and Expected", "Absent and Not Expected"])
        if valid_results: print(f"M3_Return_Documentation (Correctness): {return_correct_count}/{len(valid_results)} ({return_correct_count/len(valid_results):.2%})")
        
        acceptable_length_count = sum(1 for res in valid_results if res.get("M5_Word_Count_Readability") == "Acceptable Length")
        if valid_results: print(f"M5_Word_Count_Readability (Acceptable Length): {acceptable_length_count}/{len(valid_results)} ({acceptable_length_count/len(valid_results):.2%})")

    # Optional: Save detailed results
    # with open("batch_evaluation_results.json", "w") as f_out:
    #    json.dump(all_eval_results, f_out, indent=2)
    # print("\nFull evaluation results saved to batch_evaluation_results.json")


if __name__ == "__main__":
    # Ensure BACKEND_API_URL and BACKEND_API_KEY are correctly set at the top of this script.
    # Ensure your knowledge_base_data.json (or other dataset file) is correctly formatted.
    run_batch_evaluation(dataset_filepath="knowledge_base_data.json")