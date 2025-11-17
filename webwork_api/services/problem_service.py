"""
Problem service for business logic.

Implements the Service pattern for problem-related operations.
"""

from typing import Optional, List
import traceback

from pg.translator.translator import PGTranslator, ProblemResult

from ..models.domain import ProblemMetadata, ProblemContent, AnswerBlank, DifficultyLevel
from ..repositories.problem_repository import ProblemRepositoryInterface
from ..core.errors import ProblemNotFoundError, ProblemRenderError
from ..core.logging import get_logger

logger = get_logger(__name__)


class ProblemService:
    """
    Service for problem operations.

    Coordinates between repository and translator to provide problem functionality.
    """

    def __init__(
        self,
        repository: ProblemRepositoryInterface,
        translator: Optional[PGTranslator] = None
    ):
        self.repository = repository
        self.translator = translator or PGTranslator()

        logger.info("ProblemService initialized")

    async def get_problem_metadata(self, problem_id: str) -> ProblemMetadata:
        """Get problem metadata"""
        logger.debug(
            "Fetching problem metadata",
            extra_data={"problem_id": problem_id}
        )

        return await self.repository.get(problem_id)

    async def get_problem(self, problem_id: str, seed: int = 12345) -> ProblemContent:
        """
        Get rendered problem.

        Args:
            problem_id: Problem identifier
            seed: Random seed for problem generation

        Returns:
            Rendered problem content

        Raises:
            ProblemNotFoundError: If problem doesn't exist
            ProblemRenderError: If rendering fails
        """
        logger.info(
            "Rendering problem",
            extra_data={"problem_id": problem_id, "seed": seed}
        )

        # Get metadata
        metadata = await self.repository.get(problem_id)

        # Render problem
        try:
            result = self.translator.translate(
                metadata.file_path,
                seed=seed
            )

            # Convert to domain model
            content = self._convert_result_to_content(result, seed)

            logger.info(
                "Problem rendered successfully",
                extra_data={
                    "problem_id": problem_id,
                    "seed": seed,
                    "num_answers": len(content.answer_blanks)
                }
            )

            return content

        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(
                "Failed to render problem",
                extra_data={
                    "problem_id": problem_id,
                    "seed": seed,
                    "error": str(e),
                    "traceback": error_trace
                }
            )
            raise ProblemRenderError(problem_id, str(e))

    async def list_problems(self) -> List[ProblemMetadata]:
        """List all available problems"""
        logger.debug("Listing problems")

        problems = await self.repository.list()

        logger.info(
            "Problems listed",
            extra_data={"count": len(problems)}
        )

        return problems

    async def search_problems(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        difficulty: Optional[DifficultyLevel] = None
    ) -> List[ProblemMetadata]:
        """Search problems by criteria"""
        logger.debug(
            "Searching problems",
            extra_data={
                "query": query,
                "tags": tags,
                "difficulty": difficulty
            }
        )

        return await self.repository.search(query, tags, difficulty)

    def _convert_result_to_content(
        self,
        result: ProblemResult,
        seed: int
    ) -> ProblemContent:
        """Convert ProblemResult to ProblemContent domain model"""

        # Extract answer blanks
        answer_blanks = []
        for name, blank_info in result.answer_blanks.items():
            evaluator = (
                blank_info.get("evaluator")
                if isinstance(blank_info, dict)
                else blank_info
            )

            # Get answer type
            answer_type = "text"
            if hasattr(evaluator, "__class__"):
                answer_type = evaluator.__class__.__name__.lower()

            # Get correct answer
            correct_answer = None
            if hasattr(evaluator, "__str__"):
                correct_answer = str(evaluator)

            answer_blanks.append(
                AnswerBlank(
                    name=name,
                    type=answer_type,
                    width=20,
                    correct_answer=correct_answer,
                    options=blank_info.get("options", {}) if isinstance(blank_info, dict) else {}
                )
            )

        return ProblemContent(
            statement_html=result.statement_html,
            answer_blanks=answer_blanks,
            solution_html=result.solution_html,
            hint_html=result.hint_html,
            header_html=result.header_html,
            metadata=result.metadata or {},
            errors=result.errors,
            warnings=result.warnings,
            seed=seed
        )


# Factory function for dependency injection
def get_problem_service(
    repository: ProblemRepositoryInterface
) -> ProblemService:
    """Create problem service instance"""
    return ProblemService(repository)
