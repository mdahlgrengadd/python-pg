"""
Problem repository for data access.

Implements the Repository pattern for problem storage and retrieval.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
import json

from ..models.domain import ProblemMetadata, DifficultyLevel
from ..core.errors import ProblemNotFoundError
from ..core.logging import get_logger
from ..core.config import settings

logger = get_logger(__name__)


class ProblemRepositoryInterface(ABC):
    """Abstract interface for problem repository"""

    @abstractmethod
    async def get(self, problem_id: str) -> ProblemMetadata:
        """Get problem metadata by ID"""
        pass

    @abstractmethod
    async def list(self) -> List[ProblemMetadata]:
        """List all problems"""
        pass

    @abstractmethod
    async def search(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        difficulty: Optional[DifficultyLevel] = None
    ) -> List[ProblemMetadata]:
        """Search problems by criteria"""
        pass

    @abstractmethod
    async def exists(self, problem_id: str) -> bool:
        """Check if problem exists"""
        pass


class FileSystemProblemRepository(ProblemRepositoryInterface):
    """
    File system-based problem repository.

    Stores problems as .pg files with optional metadata in .json files.
    """

    def __init__(self, problems_dir: Optional[Path] = None):
        self.problems_dir = problems_dir or Path(settings.PROBLEMS_DIR)
        self.problems_dir.mkdir(exist_ok=True, parents=True)

        # Cache for problem metadata
        self._metadata_cache: dict[str, ProblemMetadata] = {}

        logger.info(
            "Initialized FileSystemProblemRepository",
            extra_data={"problems_dir": str(self.problems_dir)}
        )

    async def get(self, problem_id: str) -> ProblemMetadata:
        """Get problem metadata by ID"""

        # Check cache first
        if problem_id in self._metadata_cache:
            return self._metadata_cache[problem_id]

        # Find .pg file
        pg_file = self.problems_dir / f"{problem_id}.pg"

        if not pg_file.exists():
            logger.warning(
                "Problem not found",
                extra_data={"problem_id": problem_id}
            )
            raise ProblemNotFoundError(problem_id)

        # Load or create metadata
        metadata = await self._load_metadata(problem_id, pg_file)

        # Cache it
        self._metadata_cache[problem_id] = metadata

        logger.debug(
            "Problem loaded",
            extra_data={"problem_id": problem_id}
        )

        return metadata

    async def list(self) -> List[ProblemMetadata]:
        """List all problems"""

        problems = []

        for pg_file in self.problems_dir.glob("*.pg"):
            problem_id = pg_file.stem

            try:
                metadata = await self.get(problem_id)
                problems.append(metadata)
            except Exception as e:
                logger.error(
                    "Failed to load problem",
                    extra_data={
                        "problem_id": problem_id,
                        "error": str(e)
                    }
                )
                continue

        logger.info(
            "Listed problems",
            extra_data={"count": len(problems)}
        )

        return problems

    async def search(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        difficulty: Optional[DifficultyLevel] = None
    ) -> List[ProblemMetadata]:
        """Search problems by criteria"""

        all_problems = await self.list()
        results = []

        for problem in all_problems:
            # Filter by query (title or description)
            if query:
                query_lower = query.lower()
                title_match = query_lower in problem.title.lower()
                desc_match = (
                    problem.description and
                    query_lower in problem.description.lower()
                )
                if not (title_match or desc_match):
                    continue

            # Filter by tags
            if tags:
                if not any(tag in problem.tags for tag in tags):
                    continue

            # Filter by difficulty
            if difficulty and problem.difficulty != difficulty:
                continue

            results.append(problem)

        logger.info(
            "Search completed",
            extra_data={
                "query": query,
                "tags": tags,
                "difficulty": difficulty,
                "results_count": len(results)
            }
        )

        return results

    async def exists(self, problem_id: str) -> bool:
        """Check if problem exists"""
        pg_file = self.problems_dir / f"{problem_id}.pg"
        return pg_file.exists()

    async def _load_metadata(
        self,
        problem_id: str,
        pg_file: Path
    ) -> ProblemMetadata:
        """Load or create problem metadata"""

        # Check for metadata file
        metadata_file = pg_file.with_suffix(".json")

        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    data = json.load(f)
                    return ProblemMetadata(**data)
            except Exception as e:
                logger.warning(
                    "Failed to load metadata file",
                    extra_data={
                        "problem_id": problem_id,
                        "error": str(e)
                    }
                )

        # Create default metadata
        return ProblemMetadata(
            id=problem_id,
            title=self._format_title(problem_id),
            file_path=str(pg_file)
        )

    @staticmethod
    def _format_title(problem_id: str) -> str:
        """Format problem ID as title"""
        return problem_id.replace("_", " ").title()


# Singleton instance
_problem_repository: Optional[FileSystemProblemRepository] = None


def get_problem_repository() -> FileSystemProblemRepository:
    """Get problem repository instance (singleton)"""
    global _problem_repository

    if _problem_repository is None:
        _problem_repository = FileSystemProblemRepository()

    return _problem_repository
