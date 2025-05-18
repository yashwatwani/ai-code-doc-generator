# AI Code Documentation Generator - Project Log & Features

This document tracks the development progress, features implemented, and key learnings for the AI Code Documentation Generator project, aimed at building skills for an AI Engineer role.

## Current Overall Phase: Full Stack Application Development & Initial Deployment

### Core Functionality:
A web application that allows users to input code, specify the language, and receive AI-generated documentation. The system comprises a FastAPI backend (handling AI logic, authentication, rate limiting, and observability) and a Next.js frontend (providing the user interface).

---

## Implemented Features & Skills Practiced (Step-by-Step)

**Phase 1: Foundation & Core API Setup (Backend)**
*   **Step 1: Git Repository & Virtual Environment Setup**
    *   Created a Git repository (`ai-code-doc-generator`) on GitHub.
    *   Initialized a Python virtual environment (`aidoc/`) to manage dependencies.
    *   Created a comprehensive `.gitignore` file to exclude unnecessary files (virtual env, `__pycache__`, `.env` files, OS-specific files).
    *   Installed FastAPI and Uvicorn.
*   **Step 2: Basic FastAPI Application**
    *   Created `main.py` with a FastAPI app instance.
    *   Implemented a root (`/`) GET endpoint returning a welcome message.
    *   Learned to run the app locally using Uvicorn (`uvicorn main:app --reload`).
*   **Step 3: Initial Commit & `requirements.txt`**
    *   Made initial Git commits to save progress.
    *   Generated `requirements.txt` using `pip freeze > requirements.txt` to document backend dependencies.
*   **Skills:** Git, GitHub, Python, Virtual Environments, FastAPI, Uvicorn, `requirements.txt`, Basic API structure.

**Phase 2: Core API Endpoint & Data Modeling (Backend)**
*   **Step 4: Define API Endpoint for Code Documentation**
    *   Created Pydantic models (`CodeInput`, `DocumentationOutput`) for request body validation and response serialization.
    *   Implemented a `/generate-documentation/` POST endpoint in `main.py` to accept `code` and `language`.
    *   Initially returned placeholder documentation.
    *   Tested the new endpoint using FastAPI's automatic interactive API documentation (`/docs`).
*   **Step 5: Commit API Endpoint Changes**
    *   Committed the new endpoint and Pydantic model implementations.
*   **Skills:** Pydantic data validation & serialization, FastAPI POST endpoints, API design, using `/docs`.

**Phase 3: AI Integration - OpenAI (Backend)**
*   **Step 6: Integrate Basic LLM Call (OpenAI)**
    *   Signed up for OpenAI and obtained an API key.
    *   Securely managed the OpenAI API key using a `.env` file (added to `.gitignore`) and `python-dotenv` library.
    *   Installed the `openai` Python library.
    *   Updated the `/generate-documentation/` endpoint to:
        *   Construct a prompt using the input code and language.
        *   Call the OpenAI API (`gpt-3.5-turbo` model via `client.chat.completions.create`).
        *   Process the LLM's response to extract generated documentation.
        *   Handle potential errors during the API call with `try-except` blocks and `HTTPException`.
*   **Step 7: Update `requirements.txt` & Commit**
    *   Added `openai` and `python-dotenv` to `requirements.txt`.
    *   Committed AI integration changes.
*   **Skills:** Secure API key management, `python-dotenv`, `openai` SDK, Prompt Engineering (basic), Error Handling for external API calls.

**Phase 4: Testing & CI/CD Foundation (Backend)**
*   **Step 8: Add Basic Unit Tests with Pytest**
    *   Installed `pytest` and `httpx`.
    *   Created `test_main.py`.
    *   Wrote unit tests for API endpoints (`/` and `/generate-documentation/`) using FastAPI's `TestClient`.
    *   Tested success cases and error cases (e.g., missing request fields causing 422 errors).
    *   Implemented mocking for the OpenAI API client using `pytest`'s `monkeypatch` fixture to ensure tests are fast, reliable, and don't make real API calls or incur costs.
*   **Step 9: Create GitHub Actions Workflow for CI**
    *   Created `.github/workflows/python-ci.yml`.
    *   Configured the workflow to trigger on `push` and `pull_request` events to the `main` branch.
    *   Set up jobs to run tests across multiple Python versions (3.10, 3.11, 3.12) on `ubuntu-latest`.
    *   Workflow steps: Checkout code, set up Python, install dependencies from `requirements.txt`, run `pytest`.
    *   Managed dummy API keys (`OPENAI_API_KEY_DUMMY`, `MY_APP_API_KEY_FOR_CI`) as GitHub Secrets and passed them as environment variables to the test execution step in CI.
    *   Iteratively debugged and fixed CI failures related to test setup and environment variables.
*   **Skills:** `pytest`, `TestClient`, Mocking external services (`monkeypatch`), Unit testing principles, GitHub Actions, CI/CD concepts, YAML workflow configuration, GitHub Secrets, Debugging CI environments.

**Phase 5: Enhancing AI Core - Prompt Engineering (Backend)**
*   **Step 10: Refine Prompt and Test Output Structure**
    *   Iteratively improved the prompt sent to OpenAI for the `/generate-documentation/` endpoint.
    *   Added more specific formatting instructions based on the programming language (e.g., PEP 257 for Python, JSDoc for JavaScript).
    *   Included examples within the prompt to guide the LLM.
    *   Adjusted LLM parameters (e.g., `temperature=0.2`) for more deterministic and structured output.
    *   Instructed the LLM to avoid conversational fluff.
    *   Manually tested with various code snippets and languages to evaluate and refine the quality.
*   **Skills:** Advanced Prompt Engineering, LLM output control, structured prompting, few-shot examples in prompts.

**Phase 6: API Hardening - Authentication & Rate Limiting (Backend)**
*   **Step 11: Implement Basic API Key Authentication**
    *   Defined an application-specific API key (`MY_APP_API_KEY`) stored in `.env` (backend) and passed as `X-API-Key` header by clients.
    *   Created a FastAPI dependency (`get_api_key`) using `APIKeyHeader` and `Security` to validate the incoming API key.
    *   Protected the `/generate-documentation/` endpoint with this authentication dependency. The root `/` endpoint remained open.
    *   Updated unit tests (`test_main.py`) to:
        *   Include the `X-API-Key` header in requests to the protected endpoint.
        *   Test scenarios with missing, incorrect, and correct API keys.
        *   Test server-side API key misconfiguration.
    *   Updated the GitHub Actions CI workflow (`python-ci.yml`) to provide the `MY_APP_API_KEY` (via a GitHub Secret `MY_APP_API_KEY_FOR_CI`) to the test environment.
*   **Step 12: Implement Basic Rate Limiting**
    *   Installed the `slowapi` library.
    *   Initialized a `Limiter` instance in `main.py` using `get_remote_address` (IP-based) as the key function.
    *   Registered the limiter and its exception handler with the FastAPI app.
    *   Applied rate limits (e.g., `5/minute` for `/generate-documentation/`, `20/minute` for `/`) using the `@limiter.limit()` decorator.
    *   Ensured endpoint functions accept `request: Request` as a parameter for `slowapi`.
    *   Manually tested rate limiting by exceeding request limits and observing 429 "Too Many Requests" errors.
*   **Step 13: Add Unit Tests for Rate Limiting**
    *   Wrote new unit tests in `test_main.py` to verify rate limiting behavior.
    *   Tested that an endpoint allows requests up to its limit and then returns a 429 error for subsequent requests within the time window.
    *   Adapted test logic based on observed behavior of `slowapi` (Nth request triggering the limit for an "N per period" rule).
    *   Mocked the OpenAI call within the rate limit test for the documentation endpoint to isolate testing of the rate limiting mechanism.
*   **Skills:** API Security (Authentication, Rate Limiting), FastAPI Dependencies & Security Utilities, `slowapi` library, HTTP status codes (403, 429, 500), Environment variable management for different security contexts, Advanced unit testing including auth and rate limits, CI configuration for secured endpoints.

**Phase 7: Frontend Development - Initial Setup & Backend Connection**
*   **Step 14: Basic Frontend Setup with Next.js**
    *   Created a `frontend/` subdirectory for the Next.js application.
    *   Initialized a new Next.js project using `npx create-next-app@latest` with options for TypeScript, ESLint, Tailwind CSS, `src/` directory, App Router, and import alias.
    *   Successfully ran the Next.js development server (`npm run dev`) and viewed the default page.
    *   Made a minor modification to `frontend/src/app/page.tsx` to confirm editing capabilities.
*   **Step 15: Create a Basic UI for Code Input and Documentation Display**
    *   Overhauled `frontend/src/app/page.tsx` to create a dedicated UI for the application.
    *   Added UI elements: a `textarea` for code input, a `select` dropdown for language, a "Generate Documentation" button, and an area to display generated documentation or errors.
    *   Used React's `useState` hook to manage form inputs, loading state, error messages, and documentation output.
    *   Implemented a `handleSubmit` function with placeholder logic to simulate an API call and update the UI state.
*   **Step 16: Connect Frontend to FastAPI Backend**
    *   **CORS Configuration (Backend):** Added `CORSMiddleware` to `main.py` in the FastAPI backend to allow requests from the frontend's origin (e.g., `http://localhost:3001`).
    *   **Frontend API Key Management:**
        *   Created `frontend/.env.local` (gitignored) and `frontend/.env.example` (committed) for frontend-specific environment variables.
        *   Defined `NEXT_PUBLIC_BACKEND_API_KEY` (for the application's API key to access its own backend) and `NEXT_PUBLIC_BACKEND_URL` (for the backend's URL) in these files.
    *   **API Call Implementation (Frontend):**
        *   Modified `handleSubmit` in `frontend/src/app/page.tsx` to use the `fetch` API.
        *   The fetch call sends a POST request to the backend's `/generate-documentation/` endpoint.
        *   Included `Content-Type: application/json` and `X-API-Key` (from `process.env.NEXT_PUBLIC_BACKEND_API_KEY`) headers.
        *   Sent `code` and `language` state variables in the JSON request body.
        *   Implemented basic error handling for the fetch call, attempting to parse JSON error details from the backend.
        *   Updated the frontend state (`documentation`, `error`) based on the backend's response.
*   **Step 17: Enhance Frontend - Loading State and Error Display**
    *   Improved the "Generate Documentation" button to show an SVG spinner and "Generating..." text during `isLoading` state, with distinct styling.
    *   Enhanced the error display area with more prominent styling (background, border, icon) to clearly show errors from frontend validation or backend responses.
    *   Refined error message parsing in `handleSubmit` to better extract details from backend error responses.
*   **Skills:** Next.js, React, TypeScript, JSX, Tailwind CSS (basic usage via defaults), Client-side state management (`useState`), Event handling, `fetch` API, Asynchronous JavaScript (`async/await`), CORS, Frontend environment variables (`NEXT_PUBLIC_` prefix, `.env.local`), Basic UI/UX improvements.

**Phase 8: LLM Observability (Backend)**
*   **Step 18: Integrate Langfuse for LLM Observability**
    *   Signed up for Langfuse Cloud and created a project, obtaining Public and Secret Keys.
    *   Added Langfuse credentials (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`) to the backend's `.env` file.
    *   Installed the `langfuse` Python SDK.
    *   Modified `main.py`:
        *   Initialized the `Langfuse` client at application startup, ensuring it's done before OpenAI client initialization to allow for potential auto-instrumentation. Used distinct variable names (`langfuse_client_for_tracing`, `openai_llm_client`).
        *   Implemented **manual trace creation** for the `/generate-documentation/` endpoint using `langfuse_client_for_tracing.trace()`, logging `input`, `output`, `metadata`, and `tags`.
        *   Implemented **manual generation span creation** for the OpenAI call within the parent trace using `current_trace.generation()`.
        *   Logged `prompt`, `model`, `model_parameters` at the start of the generation span.
        *   Logged `output` (generated documentation) and `usage` (token counts from `completion_obj.usage`) at the end of the generation span using `generation_span.end()`.
        *   Ensured Langfuse calls are wrapped in `try-except` to prevent tracing issues from crashing the main application.
        *   Debugged and confirmed token usage data is sent to and displayed in Langfuse.
    *   Added/Updated model definitions (specifically for `gpt-3.5-turbo`) in Langfuse Project Settings for proper cost and usage tracking display.
*   **Skills:** LLM Observability concepts, Langfuse SDK, Tracing, Spans/Generations, Logging LLM inputs/outputs/metadata/token usage, Debugging SDK integrations.

**Phase 9: Deployment (Backend Live, Frontend Next)**
*   **Step 19 (Backend Part): Deploy FastAPI Backend to Render.com**
    *   Prepared backend: Ensured `requirements.txt` was up-to-date, created `Procfile` (`web: uvicorn main:app --host 0.0.0.0 --port $PORT`).
    *   Signed up for Render.com and created a new "Web Service".
    *   Connected the GitHub repository to Render.
    *   Configured the service: name, region, branch (`main`), runtime (Python), build command (`pip install -r requirements.txt`), start command (from `Procfile`).
    *   Set environment variables on Render for `OPENAI_API_KEY`, `MY_APP_API_KEY`, and Langfuse keys.
    *   Monitored deployment logs on Render.
    *   Tested the deployed backend's public URL (`/` and `/docs` endpoints).
    *   Successfully tested the protected `/generate-documentation/` endpoint on the live Render service using its `/docs` page (with `X-API-Key` authorization), and verified traces appeared in Langfuse.
    *   Updated local frontend (`page.tsx` and `frontend/.env.local`) to use the deployed Render backend URL and the correct `NEXT_PUBLIC_BACKEND_API_KEY` for testing the full local-frontend-to-live-backend flow.
    *   Created `frontend/.env.example` and ensured `frontend/.env.local` is gitignored.
*   **Skills:** Cloud deployment concepts (PaaS), Render.com platform, `Procfile`, Environment variable management in cloud environments, Testing deployed APIs, Updating frontend to use live backend URLs.

---

## Next Steps / Future Enhancements Planned
*   **Step 19 (Frontend Part):** Deploy Next.js frontend to Vercel.
    *   Update backend CORS to include Vercel production URL.
*   Frontend: Further UI/UX refinements (advanced loading, syntax highlighting).
*   Backend: Caching strategies.
*   AI: Custom Evals with 5 metrics.
*   AI: Explore AI agent frameworks or RAG if scope expands.
*   Python: Deeper OOP/Async patterns through refactoring or new features.
*   Product Sense: Continuously apply.

---