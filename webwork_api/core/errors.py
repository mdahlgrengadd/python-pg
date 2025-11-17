"""
Application exceptions and error handling.

Defines custom exceptions and error handlers for consistent error responses.
"""

from typing import Any, Dict, Optional
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from .logging import get_logger

logger = get_logger(__name__)


# Custom Exceptions

class WebWorkError(Exception):
    """Base exception for WebWork errors"""

    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ProblemNotFoundError(WebWorkError):
    """Raised when a problem is not found"""

    def __init__(self, problem_id: str):
        super().__init__(
            message=f"Problem '{problem_id}' not found",
            status_code=status.HTTP_404_NOT_FOUND,
            details={"problem_id": problem_id}
        )


class ProblemRenderError(WebWorkError):
    """Raised when a problem fails to render"""

    def __init__(self, problem_id: str, error: str):
        super().__init__(
            message=f"Failed to render problem '{problem_id}': {error}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={"problem_id": problem_id, "error": error}
        )


class GradingError(WebWorkError):
    """Raised when answer grading fails"""

    def __init__(self, problem_id: str, error: str):
        super().__init__(
            message=f"Failed to grade answers for '{problem_id}': {error}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={"problem_id": problem_id, "error": error}
        )


class ValidationError(WebWorkError):
    """Raised for validation errors"""

    def __init__(self, message: str, field: Optional[str] = None):
        details = {"field": field} if field else {}
        super().__init__(
            message=message,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details=details
        )


class RateLimitError(WebWorkError):
    """Raised when rate limit is exceeded"""

    def __init__(self):
        super().__init__(
            message="Rate limit exceeded. Please try again later.",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS
        )


class AuthenticationError(WebWorkError):
    """Raised for authentication failures"""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED
        )


class AuthorizationError(WebWorkError):
    """Raised for authorization failures"""

    def __init__(self, message: str = "Permission denied"):
        super().__init__(
            message=message,
            status_code=status.HTTP_403_FORBIDDEN
        )


# Error Response Models

def create_error_response(
    error: Exception,
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    include_details: bool = True
) -> JSONResponse:
    """Create standardized error response"""

    error_data = {
        "error": {
            "type": error.__class__.__name__,
            "message": str(error),
        }
    }

    # Add details for WebWork errors
    if isinstance(error, WebWorkError) and include_details:
        error_data["error"]["details"] = error.details

    # Log error
    logger.error(
        f"Error occurred: {error}",
        extra_data={
            "error_type": error.__class__.__name__,
            "status_code": status_code,
            **(error.details if isinstance(error, WebWorkError) else {})
        },
        exc_info=True
    )

    return JSONResponse(
        status_code=status_code,
        content=error_data
    )


# Exception Handlers

async def webwork_error_handler(request: Request, exc: WebWorkError) -> JSONResponse:
    """Handle WebWorkError exceptions"""
    return create_error_response(exc, exc.status_code)


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle HTTPException exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": "HTTPException",
                "message": exc.detail,
            }
        }
    )


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle request validation errors"""
    logger.warning(
        "Validation error",
        extra_data={"errors": exc.errors()}
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "type": "ValidationError",
                "message": "Request validation failed",
                "details": exc.errors()
            }
        }
    )


async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected errors"""
    logger.exception(
        "Unexpected error occurred",
        extra_data={"path": request.url.path}
    )

    # Don't expose internal errors in production
    from .config import settings
    include_details = settings.DEBUG

    message = str(exc) if include_details else "An internal error occurred"

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "type": "InternalServerError",
                "message": message,
            }
        }
    )


# Register all error handlers
def register_error_handlers(app):
    """Register error handlers with FastAPI app"""
    app.add_exception_handler(WebWorkError, webwork_error_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(Exception, generic_error_handler)
