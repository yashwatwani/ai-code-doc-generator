// src/app/page.tsx
'use client';

import { useState } from 'react';

// Access the environment variable
// It will be undefined if not set or not prefixed with NEXT_PUBLIC_
const YOUR_BACKEND_API_KEY = process.env.NEXT_PUBLIC_BACKEND_API_KEY; 

export default function Home() {
  const [code, setCode] = useState<string>('');
  const [language, setLanguage] = useState<string>('python');
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
    
    if (!YOUR_BACKEND_API_KEY) { // Check if the env var was loaded
        setError("Frontend API Key environment variable (NEXT_PUBLIC_BACKEND_API_KEY) is not configured. Please check your frontend/.env.local file.");
        setIsLoading(false);
        return;
    }

    try {
      const response = await fetch('http://localhost:8000/generate-documentation/', { 
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': YOUR_BACKEND_API_KEY, 
        },
        body: JSON.stringify({ code, language }),
      });

      // ... (rest of your try-catch block for handling response) ...
      if (!response.ok) {
        let errorDetail = `HTTP error! Status: ${response.status}`;
        try {
            const errorData = await response.json();
            errorDetail = errorData.detail || JSON.stringify(errorData) || errorDetail;
        } catch (e) {
            errorDetail = response.statusText || errorDetail;
        }
        throw new Error(errorDetail);
      }
      const data = await response.json();
      setDocumentation(data.generated_documentation);

    } catch (err: any) {
      console.error("API Call Error:", err);
      setError(err.message || 'Failed to generate documentation. Check console for details.');
      setDocumentation('');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="flex min-h-screen flex-col items-center justify-start p-8 sm:p-16 bg-gray-50">
      <div className="w-full max-w-3xl bg-white p-8 rounded-xl shadow-xl">
        <h1 className="text-3xl font-bold mb-8 text-center text-gray-800">
          AI Code Documentation Generator
        </h1>

        {/* Code Input Area */}
        <div className="mb-6">
          <label htmlFor="codeInput" className="block text-lg font-medium text-gray-700 mb-2">
            Paste Your Code:
          </label>
          <textarea
            id="codeInput"
            rows={10}
            className="w-full p-3 border border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 text-sm font-mono bg-gray-50 text-gray-800"
            value={code}
            onChange={(e) => setCode(e.target.value)}
            placeholder="Enter your code snippet here..."
          />
        </div>

        {/* Language Selection */}
        <div className="mb-6">
          <label htmlFor="languageSelect" className="block text-lg font-medium text-gray-700 mb-2">
            Select Language:
          </label>
          <select
            id="languageSelect"
            className="w-full p-3 border border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 text-sm bg-white text-gray-800"
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
          >
            <option value="python">Python</option>
            <option value="javascript">JavaScript</option>
            <option value="typescript">TypeScript</option>
            {/* Add more languages as needed */}
          </select>
        </div>

        {/* Submit Button */}
        <button
          onClick={handleSubmit}
          disabled={isLoading}
          className="w-full bg-indigo-600 text-white py-3 px-6 rounded-md text-lg font-semibold hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:bg-indigo-300 transition duration-150 ease-in-out"
        >
          {isLoading ? 'Generating...' : 'Generate Documentation'}
        </button>

        {/* Error Display */}
        {error && (
          <div className="mt-6 p-4 bg-red-100 text-red-700 border border-red-300 rounded-md">
            <p className="font-semibold">Error:</p>
            <p>{error}</p>
          </div>
        )}

        {/* Documentation Output Area */}
        {documentation && !error && (
          <div className="mt-8">
            <h2 className="text-2xl font-semibold text-gray-700 mb-4">
              Generated Documentation:
            </h2>
            <pre className="bg-gray-100 p-4 rounded-md overflow-x-auto text-sm font-mono text-gray-800 whitespace-pre-wrap">
              {documentation}
            </pre>
          </div>
        )}
      </div>
    </main>
  );
}