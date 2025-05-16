# test_main.py

from fastapi.testclient import TestClient
from main import app # Import your FastAPI app instance
import pytest # Import pytest to use fixtures like monkeypatch
# We need to import the actual class/module we intend to patch if patching at the class/module level
from openai.resources.chat import completions as openai_chat_completions


client = TestClient(app)

def test_read_root():
    """Test the root GET endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the AI Code Documentation Generator API"}

def test_generate_documentation_success(monkeypatch):
    """
    Test the /generate-documentation/ POST endpoint for a successful case.
    This test will MOCK the OpenAI API call.
    """

    class MockChoice:
        def __init__(self, content):
            self.message = MockMessage(content)

    class MockMessage:
        def __init__(self, content):
            self.content = content

    class MockCompletion:
        def __init__(self, content="Mocked AI Documentation"):
            self.choices = [MockChoice(content)]

    def mock_openai_completions_create(*args, **kwargs):
        print("Mocked OpenAI completions.create called") # For debugging
        return MockCompletion(content="""/**
 * Adds two numbers.
 * @param {number} a - The first number.
 * @param {number} b - The second number.
 * @returns {number} The sum of a and b.
 */""")

    # Patch the 'create' method of the 'Completions' class (or its instance)
    # The OpenAI client instance in main.py is `client`.
    # `client.chat.completions` is an instance of the Completions class.
    # So we want to patch the `create` method on that instance.
    # The most reliable way can be to patch it on the class itself, which affects all instances.
    # Or, ensure main.client is robustly available.
    
    # Let's try patching the specific method on the imported module's class if `main.client` is problematic.
    # Path: openai.resources.chat.completions.Completions.create
    monkeypatch.setattr(openai_chat_completions.Completions, "create", mock_openai_completions_create)
    # This patches the 'create' method on the Completions class itself.
    # Any call to client.chat.completions.create() in main.py should now use the mock.

    test_payload = {
        "code": "function add(a, b) { return a + b; }",
        "language": "javascript"
    }

    response = client.post("/generate-documentation/", json=test_payload)

    assert response.status_code == 200 # This was failing with 500
    response_data = response.json()
    assert response_data["message"] == "Documentation generated successfully."
    assert response_data["original_code"] == test_payload["code"]
    assert "Adds two numbers" in response_data["generated_documentation"]


def test_generate_documentation_missing_code():
    """Test the /generate-documentation/ POST endpoint when 'code' field is missing."""
    response = client.post("/generate-documentation/", json={"language": "python"}) # Missing 'code'
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