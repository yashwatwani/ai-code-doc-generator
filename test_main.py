import os
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from main import app # Only import the app for TestClient
import pytest
from openai.resources.chat import completions as openai_chat_completions
import main as main_module # Import main as a module to access/patch its globals

# Load .env for local test runs to pick up MY_APP_API_KEY for VALID_TEST_API_KEY
# and OPENAI_API_KEY for the app's client initialization (though mocked for some tests).
load_dotenv() 

# Test Client Setup
client = TestClient(app)

# This key is used by tests when sending requests requiring valid auth.
# It reads MY_APP_API_KEY from the environment, same as main.py does for EXPECTED_API_KEY.
# Ensures tests align with how the app itself determines the expected key.
VALID_TEST_API_KEY = os.getenv("MY_APP_API_KEY")

# --- Test Functions ---

def test_read_root():
    """Test the root GET endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the AI Code Documentation Generator API"}

# --- Tests for /generate-documentation/ ---

def test_generate_documentation_success_with_auth(monkeypatch):
    """
    Test successful documentation generation with valid API key.
    Mocks OpenAI API call.
    """
    if not VALID_TEST_API_KEY:
        pytest.skip("MY_APP_API_KEY is not set in the environment; skipping auth-dependent test.")

    class MockChoice:
        def __init__(self, content): self.message = MockMessage(content)
    class MockMessage:
        def __init__(self, content): self.content = content
    class MockCompletion:
        def __init__(self, content="Mocked AI Documentation"): self.choices = [MockChoice(content)]

    def mock_openai_completions_create(*args, **kwargs):
        # print("Mocked OpenAI completions.create called") # Keep for debugging if needed
        return MockCompletion(content="""/** Mocked JSDoc for test */""")

    monkeypatch.setattr(openai_chat_completions.Completions, "create", mock_openai_completions_create)

    headers = {"X-API-Key": VALID_TEST_API_KEY}
    test_payload = {"code": "function test() {}", "language": "javascript"}
    response = client.post("/generate-documentation/", json=test_payload, headers=headers)

    assert response.status_code == 200, f"Response: {response.text}"
    response_data = response.json()
    assert response_data["message"] == "Documentation generated successfully."
    assert "Mocked JSDoc for test" in response_data["generated_documentation"]

def test_generate_documentation_no_api_key():
    """Test request to protected endpoint without API key."""
    test_payload = {"code": "function test() {}", "language": "javascript"}
    response = client.post("/generate-documentation/", json=test_payload)
    assert response.status_code == 403
    assert response.json().get("detail") == "Not authenticated: X-API-Key header missing."

def test_generate_documentation_wrong_api_key():
    """Test request to protected endpoint with an incorrect API key."""
    if not VALID_TEST_API_KEY: 
        pytest.skip("MY_APP_API_KEY is not set; cannot reliably test 'wrong key' if no 'correct' key is defined.")

    headers = {"X-API-Key": "THIS_IS_A_VERY_WRONG_KEY_12345"}
    test_payload = {"code": "function test() {}", "language": "javascript"}
    response = client.post("/generate-documentation/", json=test_payload, headers=headers)
    assert response.status_code == 403
    assert response.json().get("detail") == "Could not validate credentials"

def test_generate_documentation_server_key_not_configured(monkeypatch):
    """
    Test scenario where the server's EXPECTED_API_KEY is not configured (is None).
    """
    # Store original value of main_module.EXPECTED_API_KEY if it exists
    # The `object()` is a sentinel to detect if the attribute existed at all.
    original_expected_key_value = getattr(main_module, 'EXPECTED_API_KEY', object())
    
    # Temporarily set the EXPECTED_API_KEY global in the main module to None
    # This simulates the state where os.getenv("MY_APP_API_KEY") returned None in main.py
    main_module.EXPECTED_API_KEY = None
    
    # We must send an X-API-Key header to pass the initial check in get_api_key
    # (the `if api_key_header is None:` check). The value of this header doesn't matter here.
    headers = {"X-API-Key": "any_key_will_do_as_header_must_be_present"}
    test_payload = {"code": "function test() {}", "language": "javascript"}
    
    response = client.post("/generate-documentation/", json=test_payload, headers=headers)
    
    # Restore the original value to avoid affecting other tests
    if original_expected_key_value is not object(): # If it was a real attribute
        main_module.EXPECTED_API_KEY = original_expected_key_value
    else: # It was not an attribute before our test set it to None, so try to remove it.
          # This case should ideally not happen if main.py always defines EXPECTED_API_KEY.
        if hasattr(main_module, 'EXPECTED_API_KEY'):
             delattr(main_module, 'EXPECTED_API_KEY') # Clean up if we created it

    assert response.status_code == 500
    assert response.json().get("detail") == "API Key not configured on server."


def test_generate_documentation_missing_code_with_auth():
    """Test validation error (missing code field) even with valid API key."""
    if not VALID_TEST_API_KEY:
        pytest.skip("MY_APP_API_KEY is not set in the environment; skipping auth-dependent test.")

    headers = {"X-API-Key": VALID_TEST_API_KEY}
    # Sending request missing the 'code' field
    response = client.post("/generate-documentation/", json={"language": "python"}, headers=headers) 
    
    assert response.status_code == 422 # Pydantic validation error
    response_data = response.json()
    assert "detail" in response_data
    found_code_error = False
    for error_item in response_data.get("detail", []):
        loc = error_item.get("loc", [])
        # msg = error_item.get("msg", "") # msg can vary slightly
        error_type = error_item.get("type", "")
        # Check if 'code' is in the location and type indicates a missing field
        if "code" in loc and "missing" in error_type.lower():
            found_code_error = True
            break
    assert found_code_error, f"Error detail for missing 'code' field not found in {response_data.get('detail')}"