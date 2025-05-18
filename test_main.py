import os
import time 
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from main import app # Only import the app for TestClient
import pytest
from openai.resources.chat import completions as openai_chat_completions
import main as main_module # Import main as a module to access/patch its globals

load_dotenv() 

client = TestClient(app)
# This key is used by tests when sending requests requiring valid auth.
# It reads MY_APP_API_KEY from the environment, same as main.py does for its EXPECTED_API_KEY.
VALID_TEST_API_KEY = os.getenv("MY_APP_API_KEY")

# --- Helper for Rate Limit Tests ---
DOCS_ENDPOINT_LIMIT_COUNT = 5
ROOT_ENDPOINT_LIMIT_COUNT = 20
# DOCS_ENDPOINT_LIMIT_SECONDS and ROOT_ENDPOINT_LIMIT_SECONDS are not used unless time.sleep is uncommented

# --- Test Functions ---
def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the AI Code Documentation Generator API"}

def test_generate_documentation_success_with_auth(monkeypatch):
    if not VALID_TEST_API_KEY:
        pytest.skip("MY_APP_API_KEY is not set in the environment; skipping auth-dependent test.")

    class MockChoice:
        def __init__(self, content): self.message = MockMessage(content)
    class MockMessage:
        def __init__(self, content): self.content = content
    class MockCompletion:
        def __init__(self, content="Mocked AI Documentation"): self.choices = [MockChoice(content)]

    def mock_openai_completions_create(*args, **kwargs):
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
    test_payload = {"code": "function test() {}", "language": "javascript"}
    response = client.post("/generate-documentation/", json=test_payload)
    assert response.status_code == 403
    assert response.json().get("detail") == "Not authenticated: X-API-Key header missing."

def test_generate_documentation_wrong_api_key():
    if not VALID_TEST_API_KEY: 
        pytest.skip("MY_APP_API_KEY is not set; cannot reliably test 'wrong key' if no 'correct' key is defined.")

    headers = {"X-API-Key": "THIS_IS_A_VERY_WRONG_KEY_12345"}
    test_payload = {"code": "function test() {}", "language": "javascript"}
    response = client.post("/generate-documentation/", json=test_payload, headers=headers)
    assert response.status_code == 403
    assert response.json().get("detail") == "Could not validate credentials"

def test_generate_documentation_server_key_not_configured(): # Removed monkeypatch, direct attr mod
    # Store original value of main_module.EXPECTED_API_KEY
    original_value = getattr(main_module, 'EXPECTED_API_KEY', object()) 
    
    # Temporarily set the EXPECTED_API_KEY global in the main module to None
    main_module.EXPECTED_API_KEY = None
    
    headers = {"X-API-Key": "any_key_will_do_as_header_must_be_present"}
    test_payload = {"code": "function test() {}", "language": "javascript"}
    response = client.post("/generate-documentation/", json=test_payload, headers=headers)
    
    # Restore the original value
    if original_value is not object():
        main_module.EXPECTED_API_KEY = original_value
    # No explicit delattr needed if we are sure EXPECTED_API_KEY is always defined in main.py at module level
    # If it might not be, more careful restoration would be needed. For now, this assumes it's always defined.

    assert response.status_code == 500
    assert response.json().get("detail") == "API Key authentication is not configured correctly on the server."

def test_generate_documentation_missing_code_with_auth():
    if not VALID_TEST_API_KEY:
        pytest.skip("MY_APP_API_KEY is not set in the environment; skipping auth-dependent test.")

    headers = {"X-API-Key": VALID_TEST_API_KEY}
    response = client.post("/generate-documentation/", json={"language": "python"}, headers=headers) 
    
    assert response.status_code == 422
    response_data = response.json()
    assert "detail" in response_data
    found_code_error = False
    for error_item in response_data.get("detail", []):
        loc = error_item.get("loc", [])
        error_type = error_item.get("type", "")
        if "code" in loc and "missing" in error_type.lower():
            found_code_error = True
            break
    assert found_code_error, f"Error detail for missing 'code' field not found in {response_data.get('detail')}"

# --- Rate Limiting Tests ---
@pytest.mark.slow 
def test_rate_limit_docs_endpoint(monkeypatch): # Added monkeypatch for OpenAI mocking
    if not VALID_TEST_API_KEY:
        pytest.skip("MY_APP_API_KEY is not set; skipping rate limit test that requires auth.")

    # Mock OpenAI to prevent real calls during rate limit test
    class MockChoice:
        def __init__(self, content): self.message = MockMessage(content)
    class MockMessage:
        def __init__(self, content): self.content = content
    class MockCompletion:
        def __init__(self, content="Mocked for rate limit"): self.choices = [MockChoice(content)]
        @property # Make usage accessible like an attribute
        def usage(self): return type('Usage', (), {'prompt_tokens':10,'completion_tokens':10,'total_tokens':20})()


    def mock_openai_completions_create_for_ratelimit(*args, **kwargs):
        return MockCompletion()
    monkeypatch.setattr(openai_chat_completions.Completions, "create", mock_openai_completions_create_for_ratelimit)


    headers = {"X-API-Key": VALID_TEST_API_KEY}
    payload = {"code": "def foo(): pass", "language": "python"}

    for i in range(DOCS_ENDPOINT_LIMIT_COUNT - 1):
        response = client.post("/generate-documentation/", json=payload, headers=headers)
        assert response.status_code == 200, \
            f"Request {i+1} (expected success) failed: {response.text}"

    response_at_limit = client.post("/generate-documentation/", json=payload, headers=headers)
    assert response_at_limit.status_code == 429, \
        f"Request {DOCS_ENDPOINT_LIMIT_COUNT} (expected 429) got {response_at_limit.status_code}: {response_at_limit.text}"
    assert "rate limit exceeded" in response_at_limit.text.lower()

@pytest.mark.slow 
def test_rate_limit_root_endpoint():
    for i in range(ROOT_ENDPOINT_LIMIT_COUNT - 1):
        response = client.get("/")
        assert response.status_code == 200, \
            f"Request {i+1} (expected success) failed: {response.text}"

    response_at_limit = client.get("/")
    assert response_at_limit.status_code == 429, \
        f"Request {ROOT_ENDPOINT_LIMIT_COUNT} (expected 429) got {response_at_limit.status_code}: {response_at_limit.text}"
    assert "rate limit exceeded" in response_at_limit.text.lower()