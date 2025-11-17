import React, { useState, useEffect } from 'react';
import clsx from 'clsx';

interface AnswerInputProps {
  name: string;
  type: string;
  value: string;
  onChange: (name: string, value: string) => void;
  disabled?: boolean;
  width?: number;
  feedback?: {
    correct: boolean;
    message: string;
    score: number;
  } | null;
}

/**
 * Input component for student answers
 * Supports different input types and provides visual feedback
 */
const AnswerInput: React.FC<AnswerInputProps> = ({
  name,
  type,
  value,
  onChange,
  disabled = false,
  width = 20,
  feedback = null,
}) => {
  const [localValue, setLocalValue] = useState(value);

  useEffect(() => {
    setLocalValue(value);
  }, [value]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const newValue = e.target.value;
    setLocalValue(newValue);
    onChange(name, newValue);
  };

  const inputClasses = clsx(
    'px-3 py-2 border rounded-md focus:outline-none focus:ring-2 transition-all',
    {
      'border-gray-300 focus:ring-blue-500 focus:border-blue-500': !feedback,
      'border-green-500 bg-green-50 focus:ring-green-500': feedback?.correct,
      'border-red-500 bg-red-50 focus:ring-red-500': feedback && !feedback.correct && feedback.score === 0,
      'border-yellow-500 bg-yellow-50 focus:ring-yellow-500': feedback && !feedback.correct && feedback.score > 0,
      'opacity-50 cursor-not-allowed': disabled,
    }
  );

  // Determine input size based on width parameter
  const inputSize = width > 40 ? 'w-full' : width > 20 ? 'w-96' : 'w-64';

  return (
    <div className="inline-flex flex-col gap-1">
      {type.includes('formula') || type.includes('numeric') || type.includes('real') ? (
        <input
          type="text"
          value={localValue}
          onChange={handleChange}
          disabled={disabled}
          className={clsx(inputClasses, inputSize)}
          placeholder="Enter your answer..."
          autoComplete="off"
          spellCheck="false"
        />
      ) : type.includes('matrix') || type.includes('vector') ? (
        <textarea
          value={localValue}
          onChange={handleChange}
          disabled={disabled}
          className={clsx(inputClasses, inputSize, 'min-h-[80px] font-mono')}
          placeholder="Enter your answer (e.g., [[1,2],[3,4]])"
          autoComplete="off"
          spellCheck="false"
        />
      ) : (
        <input
          type="text"
          value={localValue}
          onChange={handleChange}
          disabled={disabled}
          className={clsx(inputClasses, inputSize)}
          placeholder="Enter your answer..."
          autoComplete="off"
        />
      )}

      {/* Feedback message */}
      {feedback && (
        <div
          className={clsx('text-sm px-2 py-1 rounded', {
            'text-green-700 bg-green-50': feedback.correct,
            'text-red-700 bg-red-50': !feedback.correct && feedback.score === 0,
            'text-yellow-700 bg-yellow-50': !feedback.correct && feedback.score > 0,
          })}
        >
          {feedback.message || (feedback.correct ? '✓ Correct!' : '✗ Incorrect')}
        </div>
      )}
    </div>
  );
};

export default AnswerInput;
