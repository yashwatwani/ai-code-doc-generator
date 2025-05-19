# run_batch_evaluation.py
import json
import requests
import time
import os # For environment variables
from dotenv import load_dotenv # To load a .env file for this script

# --- Load environment variables for this script ---
load_dotenv(dotenv_path=".env.eval") # Expects a .env.eval file

# --- Target Environment Configuration ---
# Reads from environment variables, falls back to local if not set
EVAL_TARGET_ENV = os.getenv("EVAL_TARGET_ENV", "local").lower()

if EVAL_TARGET_ENV == "render":
    BACKEND_API_URL = os.getenv("RENDER_BACKEND_URL", "https://ai-code-doc-generator.onrender.com/generate-documentation/")
    BACKEND_API_KEY = os.getenv("RENDER_BACKEND_API_KEY") # Must be set in .env.eval for render
    print(f"--- EVALUATING AGAINST RENDER BACKEND: {BACKEND_API_URL} ---")
    if not BACKEND_API_KEY:
        print("ERROR: RENDER_BACKEND_API_KEY not set in .env.eval for Render target. Exiting.")
        exit()
elif EVAL_TARGET_ENV == "local":
    BACKEND_API_URL = os.getenv("LOCAL_BACKEND_URL", "http://localhost:8000/generate-documentation/")
    BACKEND_API_KEY = os.getenv("LOCAL_BACKEND_API_KEY") # Must be set in .env.eval for local
    print(f"--- EVALUATING AGAINST LOCAL BACKEND: {BACKEND_API_URL} ---")
    if not BACKEND_API_KEY:
        print("ERROR: LOCAL_BACKEND_API_KEY not set in .env.eval for Local target. Exiting.")
        exit()
else:
    print(f"ERROR: Invalid EVAL_TARGET_ENV: '{EVAL_TARGET_ENV}'. Must be 'local' or 'render'. Exiting.")
    exit()

# --- Evaluation Metric Helper Functions & Main Evaluator (Copy from your evaluate.py) ---
# def analyze_code_structure(code: str, language: str) -> dict: ...
# ... (all your evaluation functions) ...
# def evaluate_documentation(code: str, language: str, generated_doc: str) -> dict: ...
# (Make sure these are defined here)
import re 

def analyze_code_structure(code: str, language: str) -> dict: # ... (same as before) ...
    has_params = False; has_return = False
    lang_lower = language.lower() if language else "unknown"
    if lang_lower == "python":
        if "def " in code and "(" in code and ")" in code:
            params_str = code[code.find("(") + 1 : code.find(")")]
            if params_str.strip() != "" and not (params_str.strip().lower() == "self" and "," not in params_str): has_params = True 
        if "return " in code: has_return = True
    elif lang_lower == "javascript":
        if "function " in code and "(" in code and ")" in code:
            if code[code.find("(") + 1 : code.find(")")].strip() != "": has_params = True
        if "return " in code: has_return = True
    return {"expected_params": has_params, "expected_return": has_return}
def check_presence_of_summary(documentation: str, language: str) -> bool: # ... (same as before) ...
    if not documentation or not documentation.strip(): return False
    doc_lower = documentation.lower(); section_headers = ["args:", "parameters:", "@param", "returns:", "@returns", "params:"]
    first_section_index = len(documentation)
    for header in section_headers:
        try: idx = doc_lower.index(header); first_section_index = min(first_section_index, idx)
        except ValueError: continue
    summary_part = documentation[:first_section_index].strip()
    lang_lower = language.lower() if language else "unknown"
    if lang_lower == "python" and (summary_part == '"""' or summary_part == "'''"): return False
    if lang_lower == "javascript" and summary_part == '/**': return False
    return len(summary_part.split()) > 3
def check_presence_of_parameters_section(documentation: str, language: str) -> bool: # ... (same as before) ...
    doc_lower = documentation.lower(); lang_lower = language.lower() if language else "unknown"
    if lang_lower == "python": return "args:" in doc_lower
    elif lang_lower == "javascript": return "@param" in doc_lower
    return "parameters:" in doc_lower or "args:" in doc_lower or "@param" in doc_lower
def check_presence_of_returns_section(documentation: str, language: str) -> bool: # ... (same as before) ...
    doc_lower = documentation.lower(); lang_lower = language.lower() if language else "unknown"
    if lang_lower == "python": return "returns:" in doc_lower
    elif lang_lower == "javascript": return "@returns" in doc_lower
    return "returns:" in doc_lower or "@returns" in doc_lower
def check_basic_format_adherence(documentation: str, language: str) -> bool: # ... (same as before) ...
    doc_stripped = documentation.strip(); lang_lower = language.lower() if language else "unknown"
    if lang_lower == "python": return (doc_stripped.startswith('"""') and doc_stripped.endswith('"""')) or (doc_stripped.startswith("'''") and doc_stripped.endswith("'''"))
    elif lang_lower == "javascript": return doc_stripped.startswith('/**') and doc_stripped.endswith('*/')
    return True 
def check_word_count_readability(documentation: str, min_words=10, max_words=200) -> str: # ... (same as before) ...
    cleaned_doc = documentation.strip()
    if cleaned_doc.startswith('"""') and cleaned_doc.endswith('"""'): cleaned_doc = cleaned_doc[3:-3].strip()
    elif cleaned_doc.startswith("'''") and cleaned_doc.endswith("'''"): cleaned_doc = cleaned_doc[3:-3].strip()
    elif cleaned_doc.startswith('/**') and cleaned_doc.endswith('*/'):
        cleaned_doc = cleaned_doc[3:-2].strip()
        cleaned_doc_lines = [line.strip().lstrip('*').strip() for line in cleaned_doc.split('\n')]; cleaned_doc = "\n".join(cleaned_doc_lines).strip()
    if not cleaned_doc: return "Too Short (Essentially Empty)"
    word_count = len(cleaned_doc.split())
    if word_count < min_words: return "Too Short"
    elif word_count > max_words: return "Too Long"
    return "Acceptable Length"
def evaluate_documentation(code: str, language: str, generated_doc: str) -> dict: # ... (same as before) ...
    if not generated_doc or not generated_doc.strip(): return {"M1_Summary_Present": False, "M2_Parameter_Documentation": "Not Applicable (No Doc)", "M3_Return_Documentation": "Not Applicable (No Doc)", "M4_Basic_Format_Adherence": False, "M5_Word_Count_Readability": "Not Applicable (No Doc)", "Notes": "No documentation generated."}
    lang_lower = language.lower() if language else "unknown"; code_analysis = analyze_code_structure(code, lang_lower)
    m1_summary = check_presence_of_summary(generated_doc, lang_lower)
    params_section_actually_present = check_presence_of_parameters_section(generated_doc, lang_lower)
    if code_analysis["expected_params"]: m2_param_doc = "Present and Expected" if params_section_actually_present else "Missing but Expected"
    else: m2_param_doc = "Present but Not Expected (e.g., 'Args: None')" if params_section_actually_present else "Absent and Not Expected"
    returns_section_actually_present = check_presence_of_returns_section(generated_doc, lang_lower)
    if code_analysis["expected_return"]: m3_return_doc = "Present and Expected" if returns_section_actually_present else "Missing but Expected"
    else: m3_return_doc = "Present but Not Expected (e.g., 'Returns: None')" if returns_section_actually_present else "Absent and Not Expected"
    m4_format = check_basic_format_adherence(generated_doc, lang_lower); m5_readability = check_word_count_readability(generated_doc)
    return {"M1_Summary_Present": m1_summary, "M2_Parameter_Documentation": m2_param_doc, "M3_Return_Documentation": m3_return_doc, "M4_Basic_Format_Adherence": m4_format, "M5_Word_Count_Readability": m5_readability, "Code_Analysis_Debug": code_analysis }

# --- End Evaluation Functions ---

def call_generation_api(code: str, language: str) -> str | None: # ... (same as before, using global BACKEND_API_URL, BACKEND_API_KEY) ...
    headers = {"X-API-Key": BACKEND_API_KEY, "Content-Type": "application/json"}
    payload = {"code": code, "language": language}
    try:
        print(f"  Calling API for: {language} snippet (first 70 chars): {code[:70].replace('\n', '↵ ')}...")
        response = requests.post(BACKEND_API_URL, headers=headers, json=payload, timeout=90) # Increased timeout slightly
        response.raise_for_status() 
        data = response.json()
        print("  API call successful.")
        return data.get("generated_documentation")
    except requests.exceptions.Timeout: print(f"  Error calling API: Request timed out."); return None
    except requests.exceptions.HTTPError as http_err: print(f"  Error calling API: HTTP error: {http_err} - Response: {response.text[:200]}"); return None
    except requests.exceptions.RequestException as req_err: print(f"  Error calling API: {req_err}"); return None
    except json.JSONDecodeError: print(f"  Error decoding JSON. Status: {response.status_code}, Response: {response.text[:200]}"); return None

def run_batch_evaluation(dataset_filepath="knowledge_base_data.json"): # ... (same as before) ...
    try:
        with open(dataset_filepath, 'r') as f: evaluation_dataset = json.load(f)
        print(f"Loaded {len(evaluation_dataset)} items from {dataset_filepath}")
    except Exception as e: print(f"Error loading dataset '{dataset_filepath}': {e}"); return
    all_eval_results = []; successful_generations = 0
    for i, item in enumerate(evaluation_dataset):
        item_id = item.get('id', f'item_{i+1}'); item_lang = item.get('language'); item_code = item.get('code_snippet')
        if not item_code or not item_lang: print(f"\n--- Skipping Item {i+1}: {item_id} - Missing 'code_snippet' or 'language' ---"); all_eval_results.append({"id": item_id, "error": "Missing data"}); continue
        print(f"\n--- Evaluating Item {i+1}/{len(evaluation_dataset)}: {item_id} ({item_lang}) ---")
        generated_doc = call_generation_api(item_code, item_lang)
        if generated_doc is not None:
            successful_generations += 1; eval_metrics = evaluate_documentation(item_code, item_lang, generated_doc)
            print(f"  Metrics: {eval_metrics}"); all_eval_results.append({"id": item_id, "language": item_lang, "code_snippet_preview": item_code[:100] + "...", "generated_doc_preview": generated_doc[:100] + "...", **eval_metrics})
        else: print("  Failed to generate doc for this item."); all_eval_results.append({"id": item_id, "language": item_lang, "error": "API fail/empty doc"})
        if i < len(evaluation_dataset) - 1: print("  Pausing (12s) for rate limits..."); time.sleep(12)
    print("\n\n--- Overall Batch Evaluation Summary ---") # ... (aggregation logic as before) ...
    num_total_items = len(evaluation_dataset); print(f"Total items: {num_total_items}"); print(f"Successful generations: {successful_generations}/{num_total_items}")
    if successful_generations > 0:
        valid_results = [res for res in all_eval_results if "error" not in res and res.get("M1_Summary_Present") is not None]
        if valid_results:
            for metric_key, display_name, success_categories in [
                ("M1_Summary_Present", "Summary Present", [True]),
                ("M4_Basic_Format_Adherence", "Format Adherence", [True]),
                ("M2_Parameter_Documentation", "Parameter Doc Correctness", ["Present and Expected", "Absent and Not Expected"]),
                ("M3_Return_Documentation", "Return Doc Correctness", ["Present and Expected", "Absent and Not Expected"]),
                ("M5_Word_Count_Readability", "Readability (Acceptable Length)", ["Acceptable Length"])
            ]:
                count = sum(1 for res in valid_results if res.get(metric_key) in success_categories)
                print(f"{display_name}: {count}/{len(valid_results)} ({count/len(valid_results):.2%})")
        else: print("No valid results to aggregate.")

if __name__ == "__main__":
    print("Starting batch evaluation script...")
    # To run against local: set EVAL_TARGET_ENV=local (or leave unset)
    # To run against render: set EVAL_TARGET_ENV=render
    # And ensure corresponding _API_KEY and _URL are in .env.eval
    run_batch_evaluation(dataset_filepath="knowledge_base_data.json")