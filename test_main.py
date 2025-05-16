# test_main.py

from fastapi.testclient import TestClient
from main import app # Import your FastAPI app instance
import pytest # Import pytest to use fixtures like monkeypatch

# Create a TestClient instance using your FastAPI app
# This client allows you to make requests to your app in tests
# without running a live Uvicorn server.
client = TestClient(app)

def test_read_root():
    """Test the root GET endpoint."""
    response = client.get("/")
    assert response.status_code == 200 # Check if the status code is 200 OK
    assert response.json() == {"message": "Welcome to the AI Code Documentation Generator API"} # Check the JSON response

def test_generate_documentation_success(monkeypatch):
    """
    Test the /generate-documentation/ POST endpoint for a successful case.
    This test will MOCK the OpenAI API call to avoid making real API calls
    and incurring costs during testing.
    """

    # 1. Define a mock function that simulates a successful OpenAI API response
    class MockChoice:
        def __init__(self, content):
            self.message = MockMessage(content)

    class MockMessage:
        def __init__(self, content):
            self.content = content

    class MockCompletion:
        def __init__(self, content="Mocked AI Documentation"):
            self.choices = [MockChoice(content)]

    def mock_openai_create(*args, **kwargs):
        # This function will replace the actual client.chat.completions.create
        print("Mocked OpenAI create called") # For debugging, to see if mock is used
        return MockCompletion(content="""/**
 * Adds two numbers.
 * @param {number} a - The first number.
 * @param {number} b - The second number.
 * @returns {number} The sum of a and b.
 */""")

    # 2. Use monkeypatch to replace the actual OpenAI API call with our mock function
    # We need to tell monkeypatch the full path to the 'create' method we want to mock
    # Assuming 'client' is the OpenAI client instance in your 'main.py'
    # and it's accessible as 'main.client' from this test file.
    # If your OpenAI client is initialized differently or has a different name
    # in main.py, adjust the path accordingly.
    # The path would be 'module_name.openai_client_instance_name.chat.completions.create'
    if hasattr(app, 'openapi_client') and app.openapi_client is not None: # Check if client exists
        monkeypatch.setattr("main.client.chat.completions.create", mock_openai_create)
    else:
        # If the client isn't initialized (e.g. API key missing during test setup),
        # we can't easily mock it this way. We might skip this part of the test
        # or ensure a dummy client is always present in main.py for testability.
        # For now, let's print a warning if the client isn't found for mocking.
        print("Warning: OpenAI client in main.py not found or not initialized for mocking.")


    # 3. Define the test input
    test_payload = {
        "code": "function add(a, b) { return a + b; }",
        "language": "javascript"
    }

    # 4. Make the POST request to the endpoint
    response = client.post("/generate-documentation/", json=test_payload)

    # 5. Assert the response
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["message"] == "Documentation generated successfully."
    assert response_data["original_code"] == test_payload["code"]
    # Check if the generated documentation contains expected parts from the mock
    assert "Adds two numbers" in response_data["generated_documentation"]


# test_main.py

# ... (other imports and client setup) ...

def test_generate_documentation_missing_code():
    """Test the /generate-documentation/ POST endpoint when 'code' field is missing."""
    response = client.post("/generate-documentation/", json={"language": "python"}) # Missing 'code'
    assert response.status_code == 422 # FastAPI should return 422 Unprocessable Entity
    response_data = response.json()
    
    # ---- START DEBUG PRINT ----
    # print("\nDEBUG: response_data['detail'] for missing code:")
    # import json
    # print(json.dumps(response_data.get("detail"), indent=2))
    # ---- END DEBUG PRINT ----
    
    assert "detail" in response_data
    
    found_code_error = False
    for error_item in response_data.get("detail", []): # Use .get for safety
        # Check if 'loc' exists and 'code' is in the second element of 'loc' (usually ['body', 'field_name'])
        loc = error_item.get("loc", [])
        msg = error_item.get("msg", "")
        error_type = error_item.get("type", "") # Get the error type as well

        # More robust check:
        # Check if 'code' is one of the field names in the location path
        # And if the message indicates a required/missing field.
        # Pydantic v2 uses "type": "missing" for required fields.
        if "code" in loc and ("missing" in error_type.lower() or "field required" in msg.lower()):
            found_code_error = True
            break
    assert found_code_error, "Error detail for missing 'code' field not found"

# ... (other tests) ..
# You might need to adjust the main.py to make the OpenAI client instance
# more easily accessible for mocking, e.g., by attaching it to the app instance
# or having a clear global variable.
# For example, in main.py, after client = OpenAI(), you could do:
# app.openapi_client = client
# Then in tests: monkeypatch.setattr(app.openapi_client.chat.completions, "create", mock_openai_create)
# Let's assume for now `main.client` is how the OpenAI client is named and accessible.
# If you named your OpenAI client instance differently in main.py,
# adjust `main.client.chat.completions.create` path in monkeypatch.setattr accordingly.