export interface AnswerBlankInfo {
  name: string;
  type: string;
  width?: number;
  correct_answer?: string | null;
}

export interface ProblemResponse {
  statement_html: string;
  answer_blanks: AnswerBlankInfo[];
  solution_html?: string | null;
  hint_html?: string | null;
  header_html?: string | null;
  metadata?: Record<string, any> | null;
  errors?: string[] | null;
  warnings?: string[] | null;
  seed: number;
}

export interface AnswerResultResponse {
  score: number;
  correct: boolean;
  student_answer: string;
  student_correct_answer: string;
  answer_message: string;
  messages: string[];
  type: string;
  preview: string;
  error_message: string;
  error_flag: boolean;
  ans_label: string;
}

export interface GradeResponse {
  score: number;
  answer_results: Record<string, AnswerResultResponse>;
  problem_result?: Record<string, any> | null;
  problem_state?: Record<string, any> | null;
}

export interface GradeRequest {
  answers: Record<string, string>;
  seed?: number;
}
