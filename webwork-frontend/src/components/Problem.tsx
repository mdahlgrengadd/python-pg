import React, { useState, useEffect } from 'react';
import { ProblemResponse, GradeResponse, AnswerResultResponse } from '../types';
import { problemsApi } from '../api';
import HTMLRenderer from './HTMLRenderer';
import AnswerInput from './AnswerInput';
import clsx from 'clsx';

interface ProblemProps {
  problemId: string;
  seed?: number;
}

/**
 * Main problem component that handles:
 * - Loading and displaying problems
 * - Managing answer inputs
 * - Submitting answers for grading
 * - Displaying feedback
 */
const Problem: React.FC<ProblemProps> = ({ problemId, seed = 12345 }) => {
  const [problem, setProblem] = useState<ProblemResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [grading, setGrading] = useState(false);
  const [gradeResult, setGradeResult] = useState<GradeResponse | null>(null);
  const [showSolution, setShowSolution] = useState(false);
  const [showHint, setShowHint] = useState(false);

  useEffect(() => {
    loadProblem();
  }, [problemId, seed]);

  const loadProblem = async () => {
    try {
      setLoading(true);
      setError(null);
      setGradeResult(null);
      const data = await problemsApi.get(problemId, seed);
      setProblem(data);

      // Initialize answers object
      const initialAnswers: Record<string, string> = {};
      data.answer_blanks.forEach(blank => {
        initialAnswers[blank.name] = '';
      });
      setAnswers(initialAnswers);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to load problem');
    } finally {
      setLoading(false);
    }
  };

  const handleAnswerChange = (name: string, value: string) => {
    setAnswers(prev => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    try {
      setGrading(true);
      const result = await problemsApi.grade(problemId, {
        answers,
        seed,
      });
      setGradeResult(result);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to grade answers');
    } finally {
      setGrading(false);
    }
  };

  const handleReset = () => {
    setGradeResult(null);
    const resetAnswers: Record<string, string> = {};
    problem?.answer_blanks.forEach(blank => {
      resetAnswers[blank.name] = '';
    });
    setAnswers(resetAnswers);
    setShowSolution(false);
    setShowHint(false);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading problem...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-6">
        <h3 className="text-red-800 font-semibold mb-2">Error</h3>
        <p className="text-red-700">{error}</p>
        <button
          onClick={loadProblem}
          className="mt-4 px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors"
        >
          Retry
        </button>
      </div>
    );
  }

  if (!problem) {
    return null;
  }

  const overallScore = gradeResult?.score ?? null;
  const hasBeenGraded = gradeResult !== null;

  return (
    <div className="max-w-4xl mx-auto">
      {/* Problem Statement */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        {problem.statement_html && (
          <HTMLRenderer html={problem.statement_html} />
        )}

        {/* Answer Inputs */}
        {problem.answer_blanks.length > 0 && (
          <form onSubmit={handleSubmit} className="mt-6 space-y-4">
            {problem.answer_blanks.map((blank, index) => {
              const feedback = gradeResult?.answer_results?.[blank.name];
              return (
                <div key={blank.name} className="flex flex-col gap-2">
                  <label className="font-medium text-gray-700">
                    Answer {problem.answer_blanks.length > 1 ? index + 1 : ''}:
                  </label>
                  <AnswerInput
                    name={blank.name}
                    type={blank.type}
                    value={answers[blank.name] || ''}
                    onChange={handleAnswerChange}
                    disabled={grading}
                    width={blank.width}
                    feedback={
                      feedback
                        ? {
                            correct: feedback.correct,
                            message: feedback.answer_message,
                            score: feedback.score,
                          }
                        : null
                    }
                  />
                </div>
              );
            })}

            {/* Submit Button */}
            <div className="flex gap-3 mt-6">
              <button
                type="submit"
                disabled={grading}
                className={clsx(
                  'px-6 py-3 rounded-md font-medium transition-all',
                  grading
                    ? 'bg-gray-400 cursor-not-allowed'
                    : 'bg-blue-600 hover:bg-blue-700 text-white shadow-md hover:shadow-lg'
                )}
              >
                {grading ? 'Grading...' : 'Submit Answers'}
              </button>

              {hasBeenGraded && (
                <button
                  type="button"
                  onClick={handleReset}
                  className="px-6 py-3 bg-gray-200 text-gray-700 rounded-md font-medium hover:bg-gray-300 transition-all"
                >
                  Try Again
                </button>
              )}
            </div>
          </form>
        )}
      </div>

      {/* Overall Score */}
      {hasBeenGraded && overallScore !== null && (
        <div
          className={clsx('rounded-lg p-6 mb-6 shadow-md', {
            'bg-green-50 border-2 border-green-500': overallScore >= 1.0,
            'bg-yellow-50 border-2 border-yellow-500': overallScore > 0 && overallScore < 1.0,
            'bg-red-50 border-2 border-red-500': overallScore === 0,
          })}
        >
          <h3 className="text-lg font-semibold mb-2">
            {overallScore >= 1.0 ? '🎉 Perfect Score!' : overallScore > 0 ? '⚠️ Partial Credit' : '❌ Incorrect'}
          </h3>
          <p className="text-xl font-bold">
            Score: {(overallScore * 100).toFixed(0)}%
          </p>
        </div>
      )}

      {/* Hint */}
      {problem.hint_html && (
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <button
            onClick={() => setShowHint(!showHint)}
            className="flex items-center gap-2 text-blue-600 hover:text-blue-700 font-medium"
          >
            <span>{showHint ? '▼' : '▶'}</span>
            <span>Hint</span>
          </button>
          {showHint && (
            <div className="mt-4 pl-6 border-l-4 border-blue-500">
              <HTMLRenderer html={problem.hint_html} />
            </div>
          )}
        </div>
      )}

      {/* Solution */}
      {problem.solution_html && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <button
            onClick={() => setShowSolution(!showSolution)}
            className="flex items-center gap-2 text-green-600 hover:text-green-700 font-medium"
          >
            <span>{showSolution ? '▼' : '▶'}</span>
            <span>Solution</span>
          </button>
          {showSolution && (
            <div className="mt-4 pl-6 border-l-4 border-green-500">
              <HTMLRenderer html={problem.solution_html} />
            </div>
          )}
        </div>
      )}

      {/* Debug Info */}
      {problem.metadata && (
        <div className="mt-6 text-xs text-gray-500">
          Seed: {problem.metadata.seed} | Answers: {problem.metadata.num_answers}
        </div>
      )}
    </div>
  );
};

export default Problem;
