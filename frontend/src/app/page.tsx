// src/app/page.tsx
'use client';

import { useState } from 'react';

// Access environment variables
const BACKEND_API_KEY = process.env.NEXT_PUBLIC_BACKEND_API_KEY;
const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL;

export default function Home() {
  const [code, setCode] = useState<string>('');
  const [language, setLanguage] = useState<string>('python'); // Default language
  const [documentation, setDocumentation] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async () => {
    setIsLoading(true);
    setError(null);
    setDocumentation('');

    if (code.trim() === '') {
      setError("Please enter some code.");
      setIsLoading(false);
      return;
    }
    
    if (!BACKEND_API_KEY) {
        setError("Frontend API Key environment variable (NEXT_PUBLIC_BACKEND_API_KEY) is not configured.");
        setIsLoading(false);
        return;
    }

    if (!BACKEND_URL) {
        setError("Backend URL environment variable (NEXT_PUBLIC_BACKEND_URL) is not configured.");
        setIsLoading(false);
        return;
    }

    try {
      const endpoint = `${BACKEND_URL}/generate-documentation/`;
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': BACKEND_API_KEY, 
        },
        body: JSON.stringify({ code, language }),
      });

      const responseData = await response.json(); 

      if (!response.ok) {
        const errorDetail = responseData.detail || `HTTP error! Status: ${response.status} - ${response.statusText || 'Unknown error'}`;
        throw new Error(errorDetail);
      }
      
      setDocumentation(responseData.generated_documentation);

    } catch (error: unknown) { // Corrected typing for the catch block variable
      console.error("API Call Error:", error);
      let message = 'Failed to generate documentation. Please try again.';
      if (error instanceof Error) {
        message = error.message; 
      } else if (typeof error === 'string') {
        message = error; 
      }
      setError(message);
      setDocumentation('');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="flex min-h-screen flex-col items-center justify-start p-8 sm:p-16 bg-gray-100">
      <div className="w-full max-w-3xl bg-white p-8 rounded-xl shadow-2xl">
        <h1 className="text-4xl font-bold mb-10 text-center text-gray-800">
          AI Code Documentation Generator
        </h1>

        <div className="mb-6">
          <label htmlFor="codeInput" className="block text-lg font-medium text-gray-700 mb-2">
            Paste Your Code:
          </label>
          <textarea
            id="codeInput"
            rows={12}
            className="w-full p-3 border border-gray-300 rounded-lg shadow-sm focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 text-sm font-mono bg-gray-50 text-gray-900 resize-y"
            value={code}
            onChange={(e) => setCode(e.target.value)}
            placeholder="Enter your code snippet here..."
          />
        </div>

        <div className="mb-8">
          <label htmlFor="languageSelect" className="block text-lg font-medium text-gray-700 mb-2">
            Select Language:
          </label>
          <select
            id="languageSelect"
            className="w-full p-3 border border-gray-300 rounded-lg shadow-sm focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 text-sm bg-white text-gray-900"
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
          >
            <option value="python">Python</option>
            <option value="javascript">JavaScript</option>
            <option value="typescript">TypeScript</option>
            <option value="java">Java</option>
            <option value="csharp">C#</option>
            <option value="cpp">C++</option>
            <option value="c">C</option>
            <option value="go">Go</option>
            <option value="rust">Rust</option>
          </select>
        </div>

        <button
          onClick={handleSubmit}
          disabled={isLoading}
          className={`w-full text-white py-3.5 px-6 rounded-lg text-lg font-semibold focus:outline-none focus:ring-2 focus:ring-offset-2 transition duration-150 ease-in-out flex items-center justify-center
                      ${isLoading 
                        ? 'bg-indigo-400 cursor-not-allowed' 
                        : 'bg-indigo-600 hover:bg-indigo-700 focus:ring-indigo-500'
                      }`}
        >
          {isLoading ? (
            <>
              <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Generating...
            </>
          ) : (
            'Generate Documentation'
          )}
        </button>

        {error && (
          <div className="mt-8 p-4 bg-red-50 text-red-700 border-l-4 border-red-500 rounded-md shadow-md">
            <div className="flex">
              <div className="py-1">
                <svg className="h-6 w-6 text-red-500 mr-3" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <div>
                <p className="font-bold">An error occurred:</p>
                <p className="text-sm">{error}</p>
              </div>
            </div>
          </div>
        )}

        {documentation && !error && (
          <div className="mt-10">
            <h2 className="text-2xl font-semibold text-gray-700 mb-4">
              Generated Documentation:
            </h2>
            <pre className="bg-gray-800 text-gray-100 p-6 rounded-lg overflow-x-auto text-sm font-mono whitespace-pre-wrap shadow-inner">
              {documentation}
            </pre>
          </div>
        )}
      </div>
    </main>
  );
}