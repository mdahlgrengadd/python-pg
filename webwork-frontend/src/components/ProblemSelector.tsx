import React, { useState, useEffect } from 'react';
import { problemsApi } from '../api';

interface ProblemSelectorProps {
  selectedProblem: string | null;
  onSelect: (problemId: string) => void;
}

/**
 * Problem selector component for choosing which problem to work on
 */
const ProblemSelector: React.FC<ProblemSelectorProps> = ({ selectedProblem, onSelect }) => {
  const [problems, setProblems] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadProblems();
  }, []);

  const loadProblems = async () => {
    try {
      setLoading(false);
      const data = await problemsApi.list();
      setProblems(data);

      // Auto-select first problem if none selected
      if (!selectedProblem && data.length > 0) {
        onSelect(data[0]);
      }
    } catch (err: any) {
      setError(err.message || 'Failed to load problems');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="animate-pulse">
        <div className="h-10 bg-gray-200 rounded"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-red-600 text-sm">
        Error loading problems: {error}
      </div>
    );
  }

  if (problems.length === 0) {
    return (
      <div className="text-gray-500 text-sm">
        No problems available
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2">
      <label className="text-sm font-medium text-gray-700">
        Select a problem:
      </label>
      <select
        value={selectedProblem || ''}
        onChange={(e) => onSelect(e.target.value)}
        className="px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
      >
        {problems.map((problemId) => (
          <option key={problemId} value={problemId}>
            {problemId.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
          </option>
        ))}
      </select>
    </div>
  );
};

export default ProblemSelector;
