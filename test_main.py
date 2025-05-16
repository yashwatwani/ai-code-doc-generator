# test_main.py

import os
from fastapi.testclient import TestClient
from main import app, EXPECTED_API_KEY as MAIN_EXPECTED_API_KEY # Import your FastAPI app instance and EXPECTED_API_KEY
import pytest
from openai.resources.chat import completions as openai_chat_completions

# Test Client Setup
client = TestClient(app)

# test_main.py
# ...
TEST_USER_API_KEY = "test_dummy_app_api_key_for_pytest" # This is just for test clarity
# The actual validation will use EXPECTED_API_KEY loaded from environment by main.py

# Define a key that our tests will use. For tests to pass against the actual
# get_api_key dependency, the environment running pytest needs MY_APP_API_KEY
# to be set to this value, or we need to mock get_api_key.
# Let's assume the environment will provide it for now for integration-style testing of the auth.
# If EXPECTED_API_KEY is not set in the environment main.py runs in, main.py's get_api_key will raise 500.
# For pure unit testing of generate_docs logic, we'd mock get_api_key itself.
# For now, we test the integration WITH the auth dependency.
# We will need to ensure MY_APP_API_KEY is set in the CI environment.
# For local testing, your .env should already be setting MY_APP_API_KEY.

# Let's use the actual key expected by the app for valid test cases.
# Ensure your .env file has MY_APP_API_KEY set when running tests locally.
# If MAIN_EXPECTED_API_KEY is None (not set in .env), these tests requiring auth will behave accordingly.
VALID_TEST_API_KEY = MAIN_EXPECTED_API_KEY # Use the one loaded by main.py

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
        print("Mocked OpenAI completions.create called")
        return MockCompletion(content="""/** Mocked JSDoc */""")

    monkeypatch.setattr(openai_chat_completions.Completions, "create", mock_openai_completions_create)

    headers = {"X-API-Key": VALID_TEST_API_KEY}
    test_payload = {"code": "function test() {}", "language": "javascript"}
    response = client.post("/generate-documentation/", json=test_payload, headers=headers)

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["message"] == "Documentation generated successfully."
    assert "Mocked JSDoc" in response_data["generated_documentation"]

def test_generate_documentation_no_api_key():
    """Test request to protected endpoint without API key."""
    test_payload = {"code": "function test() {}", "language": "javascript"}
    response = client.post("/generate-documentation/", json=test_payload)
    # APIKeyHeader with auto_error=True should result in 403 if header is missing
    assert response.status_code == 403 
    assert "not authenticated" in response.json().get("detail", "").lower() # Or similar default message

def test_generate_documentation_wrong_api_key():
    """Test request to protected endpoint with an incorrect API key."""
    if not VALID_TEST_API_KEY: # If no valid key is configured, this test's premise is harder to check
        pytest.skip("MY_APP_API_KEY is not set; cannot reliably test 'wrong key' scenario this way.")

    headers = {"X-API-Key": "WRONG_KEY_12345"}
    test_payload = {"code": "function test() {}", "language": "javascript"}
    response = client.post("/generate-documentation/", json=test_payload, headers=headers)
    assert response.status_code == 403
    assert response.json().get("detail") == "Could not validate credentials"

def test_generate_documentation_server_key_not_configured(monkeypatch):
    """
    Test scenario where the server might not have EXPECTED_API_KEY configured.
    This requires monkeypatching os.getenv for EXPECTED_API_KEY within the app's context for this test.
    Alternatively, we can rely on the actual `get_api_key` to raise this.
    If we ensure MY_APP_API_KEY is *not* set in the env for *this specific test*, it would also work.
    Let's try by temporarily unsetting what main.py's EXPECTED_API_KEY would see.
    """
    # Temporarily make the app believe EXPECTED_API_KEY is not set
    monkeypatch.setattr("main.EXPECTED_API_KEY", None)
    
    # We still need to send *some* X-API-Key to get past the initial APIKeyHeader check
    headers = {"X-API-Key": "any_key_as_header_is_present"} 
    test_payload = {"code": "function test() {}", "language": "javascript"}
    response = client.post("/generate-documentation/", json=test_payload, headers=headers)
    
    assert response.status_code == 500
    assert response.json().get("detail") == "API Key not configured on server."
    
    # Important: Restore for other tests if monkeypatching a global like this,
    # though pytest typically isolates monkeypatch effects per test.
    # For module-level globals, it's safer to ensure it's restored or a fresh import is used.
    # However, since MAIN_EXPECTED_API_KEY is imported at the start, this change is isolated.

def test_generate_documentation_missing_code_with_auth():
    """Test validation error (missing code field) even with valid API key."""
    if not VALID_TEST_API_KEY:
        pytest.skip("MY_APP_API_KEY is not set in the environment; skipping auth-dependent test.")

    headers = {"X-API-Key": VALID_TEST_API_KEY}
    response = client.post("/generate-documentation/", json={"language": "python"}, headers=headers) # Missing 'code'
    assert response.status_code == 422
    response_data = response.json()
    assert "detail" in response_data
    found_code_error = False
    for error_item in response_data.get("detail", []):
        loc = error_item.get("loc", [])
        msg = error_item.get("msg", "")
        error_type = error_item.get("type", "")
        if "code" in loc and ("missing" in error_type.lower() or "field required" in msg.lower()):
            found_code_error = True
            break
    assert found_code_error, "Error detail for missing 'code' field not found"