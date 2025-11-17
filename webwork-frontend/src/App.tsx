import React, { useState } from 'react';
import Problem from './components/Problem';
import ProblemSelector from './components/ProblemSelector';

function App() {
  const [selectedProblem, setSelectedProblem] = useState<string | null>(null);
  const [seed, setSeed] = useState<number>(12345);

  const handleNewSeed = () => {
    setSeed(Math.floor(Math.random() * 100000));
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-gradient-to-r from-blue-600 to-blue-700 text-white shadow-lg">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <h1 className="text-3xl font-bold">WebWork Python</h1>
          <p className="mt-2 text-blue-100">Interactive Mathematical Problem System</p>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Controls */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <ProblemSelector
              selectedProblem={selectedProblem}
              onSelect={setSelectedProblem}
            />

            <div className="flex flex-col gap-2">
              <label className="text-sm font-medium text-gray-700">
                Random Seed:
              </label>
              <div className="flex gap-2">
                <input
                  type="number"
                  value={seed}
                  onChange={(e) => setSeed(parseInt(e.target.value) || 12345)}
                  className="flex-1 px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
                <button
                  onClick={handleNewSeed}
                  className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
                  title="Generate new random seed"
                >
                  🎲 New
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Problem Display */}
        {selectedProblem ? (
          <Problem key={`${selectedProblem}-${seed}`} problemId={selectedProblem} seed={seed} />
        ) : (
          <div className="bg-white rounded-lg shadow-md p-12 text-center">
            <div className="text-gray-400 mb-4">
              <svg
                className="mx-auto h-24 w-24"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                />
              </svg>
            </div>
            <h3 className="text-xl font-semibold text-gray-700 mb-2">
              No Problem Selected
            </h3>
            <p className="text-gray-500">
              Select a problem from the dropdown above to get started.
            </p>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="text-center text-gray-500 text-sm">
            <p>WebWork Python - Open Source Mathematical Problem System</p>
            <p className="mt-1">
              Built with React, TypeScript, Tailwind CSS, FastAPI, and Pydantic
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
