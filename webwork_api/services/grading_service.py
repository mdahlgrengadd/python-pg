"""
Grading service for answer evaluation.

Handles answer submission and grading logic.
"""

from typing import Dict
import traceback

from pg.translator.translator import PGTranslator

from ..models.domain import AnswerFeedback, SessionState
from ..repositories.problem_repository import ProblemRepositoryInterface
from ..core.errors import GradingError
from ..core.logging import get_logger

logger = get_logger(__name__)


class GradingService:
    """
    Service for answer grading operations.

    Evaluates student answers and provides feedback.
    """

    def __init__(
        self,
        repository: ProblemRepositoryInterface,
        translator: PGTranslator | None = None
    ):
        self.repository = repository
        self.translator = translator or PGTranslator()

        logger.info("GradingService initialized")

    async def grade_answers(
        self,
        problem_id: str,
        answers: Dict[str, str],
        seed: int = 12345,
        session: SessionState | None = None
    ) -> tuple[float, Dict[str, AnswerFeedback], Dict]:
        """
        Grade student answers.

        Args:
            problem_id: Problem identifier
            answers: Student answers by blank name
            seed: Random seed used for problem
            session: Optional session state

        Returns:
            Tuple of (overall_score, answer_feedback_dict, problem_result)

        Raises:
            ProblemNotFoundError: If problem doesn't exist
            GradingError: If grading fails
        """
        logger.info(
            "Grading answers",
            extra_data={
                "problem_id": problem_id,
                "seed": seed,
                "num_answers": len(answers),
                "session_id": str(session.session_id) if session else None
            }
        )

        # Get problem metadata
        metadata = await self.repository.get(problem_id)

        # Grade using translator
        try:
            result = self.translator.translate(
                metadata.file_path,
                seed=seed,
                inputs=answers
            )

            # Convert results to domain models
            feedback_dict = {}
            if result.answer_results:
                for name, ans_result in result.answer_results.items():
                    feedback_dict[name] = AnswerFeedback(
                        answer_blank_name=name,
                        score=ans_result.score,
                        correct=ans_result.correct,
                        student_answer=ans_result.student_answer,
                        correct_answer=ans_result.student_correct_answer,
                        message=ans_result.answer_message,
                        messages=ans_result.messages,
                        preview=ans_result.preview,
                        error_message=ans_result.error_message
                    )

            # Calculate overall score
            overall_score = result.score or 0.0

            # Update session if provided
            if session:
                session.record_attempt(overall_score)

            logger.info(
                "Grading completed",
                extra_data={
                    "problem_id": problem_id,
                    "overall_score": overall_score,
                    "session_id": str(session.session_id) if session else None
                }
            )

            return overall_score, feedback_dict, result.problem_result or {}

        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(
                "Failed to grade answers",
                extra_data={
                    "problem_id": problem_id,
                    "seed": seed,
                    "error": str(e),
                    "traceback": error_trace
                }
            )
            raise GradingError(problem_id, str(e))


# Factory function
def get_grading_service(
    repository: ProblemRepositoryInterface
) -> GradingService:
    """Create grading service instance"""
    return GradingService(repository)
