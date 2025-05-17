# test_main.py

import os
import time # Import time for testing time-based limits
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from main import app # Only import the app for TestClient
import pytest
from openai.resources.chat import completions as openai_chat_completions
import main as main_module 

load_dotenv() 

client = TestClient(app)
VALID_TEST_API_KEY = os.getenv("MY_APP_API_KEY")

# --- Helper for Rate Limit Tests ---
# It's good to have the limit values accessible if they are defined in main.py
# For this example, we'll hardcode them in the test based on what we set in main.py
# Or, you could import them if you define them as constants in main.py
DOCS_ENDPOINT_LIMIT_COUNT = 5
DOCS_ENDPOINT_LIMIT_SECONDS = 60 # 5 per minute
ROOT_ENDPOINT_LIMIT_COUNT = 20
ROOT_ENDPOINT_LIMIT_SECONDS = 60 # 20 per minute


# --- Existing Test Functions (Keep As Is) ---
def test_read_root():
    # ... (existing code) ...
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the AI Code Documentation Generator API"}

def test_generate_documentation_success_with_auth(monkeypatch):
    # ... (existing code) ...
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
    # ... (existing code) ...
    test_payload = {"code": "function test() {}", "language": "javascript"}
    response = client.post("/generate-documentation/", json=test_payload)
    assert response.status_code == 403
    assert response.json().get("detail") == "Not authenticated: X-API-Key header missing."


def test_generate_documentation_wrong_api_key():
    # ... (existing code) ...
    if not VALID_TEST_API_KEY: 
        pytest.skip("MY_APP_API_KEY is not set; cannot reliably test 'wrong key' if no 'correct' key is defined.")

    headers = {"X-API-Key": "THIS_IS_A_VERY_WRONG_KEY_12345"}
    test_payload = {"code": "function test() {}", "language": "javascript"}
    response = client.post("/generate-documentation/", json=test_payload, headers=headers)
    assert response.status_code == 403
    assert response.json().get("detail") == "Could not validate credentials"


def test_generate_documentation_server_key_not_configured(monkeypatch):
    # ... (existing code) ...
    original_expected_key_value = getattr(main_module, 'EXPECTED_API_KEY', object()) 
    main_module.EXPECTED_API_KEY = None
    
    headers = {"X-API-Key": "any_key_will_do_as_header_must_be_present"}
    test_payload = {"code": "function test() {}", "language": "javascript"}
    response = client.post("/generate-documentation/", json=test_payload, headers=headers)
    
    if original_expected_key_value is not object():
        main_module.EXPECTED_API_KEY = original_expected_key_value
    elif hasattr(main_module, 'EXPECTED_API_KEY'):
         delattr(main_module, 'EXPECTED_API_KEY')

    assert response.status_code == 500
    assert response.json().get("detail") == "API Key authentication is not configured correctly on the server."


def test_generate_documentation_missing_code_with_auth():
    # ... (existing code) ...
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


# --- NEW Rate Limiting Tests ---

@pytest.mark.slow # Optional: mark as slow if these tests take longer due to time.sleep
def test_rate_limit_docs_endpoint():
    """Test rate limiting on the /generate-documentation/ endpoint."""
    if not VALID_TEST_API_KEY:
        pytest.skip("MY_APP_API_KEY is not set; skipping rate limit test that requires auth.")

    headers = {"X-API-Key": VALID_TEST_API_KEY}
    payload = {"code": "def foo(): pass", "language": "python"}

    # test_main.py (inside test_rate_limit_docs_endpoint)

    # Hit the endpoint up to one less than the limit
    # If limit is 5, we make 4 successful calls
    for i in range(DOCS_ENDPOINT_LIMIT_COUNT - 1): 
        response = client.post("/generate-documentation/", json=payload, headers=headers)
        assert response.status_code == 200, f"Request {i+1} (within limit) failed unexpectedly: {response.text}"

    # The next request should be the one that hits the limit count (e.g., the 5th request)
    # Let's see if this one is allowed or rejected.
    response_at_limit = client.post("/generate-documentation/", json=payload, headers=headers)
    
    # Depending on slowapi's exact behavior (inclusive/exclusive):
    # Option A: The Nth request is still allowed
    if response_at_limit.status_code == 200:
        # Then the (N+1)th request should be blocked
        response_over_limit = client.post("/generate-documentation/", json=payload, headers=headers)
        assert response_over_limit.status_code == 429, f"Expected 429 on request {DOCS_ENDPOINT_LIMIT_COUNT + 1}"
        assert "rate limit exceeded" in response_over_limit.text.lower()
    # Option B: The Nth request is already blocked
    elif response_at_limit.status_code == 429:
        assert "rate limit exceeded" in response_at_limit.text.lower()
    else:
        pytest.fail(f"Unexpected status code {response_at_limit.status_code} on {DOCS_ENDPOINT_LIMIT_COUNT}th request. Response: {response_at_limit.text}")


@pytest.mark.slow 
def test_rate_limit_root_endpoint():
    """Test rate limiting on the / endpoint."""
    # Make N-1 successful requests (e.g., if limit is 20, make 19 successful ones)
    for i in range(ROOT_ENDPOINT_LIMIT_COUNT - 1):
        response = client.get("/")
        assert response.status_code == 200, \
            f"Request {i+1} (expected success) failed: {response.text}"

    # The Nth request (e.g., 20th request) should now trigger the limit
    response_at_limit = client.get("/")
    assert response_at_limit.status_code == 429, \
        f"Request {ROOT_ENDPOINT_LIMIT_COUNT} (expected 429) got {response_at_limit.status_code}: {response_at_limit.text}"
    assert "rate limit exceeded" in response_at_limit.text.lower()

    # Optional: Test if the limit resets after the window.
    # print(f"\nRate limit hit for /, sleeping for {ROOT_ENDPOINT_LIMIT_SECONDS}s...")
    # time.sleep(ROOT_ENDPOINT_LIMIT_SECONDS + 1)
    # response_after_wait = client.get("/")
    # assert response_after_wait.status_code == 200, \
    #    f"Rate limit did not reset after waiting. Response: {response_after_wait.text}"