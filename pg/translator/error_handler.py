"""
Error Handling System for PG Translator.

Provides formatted error messages with stack traces and file mapping.
Reference: Translator.pm:533-586
"""

from __future__ import annotations

import re
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .executor import PGEnvironment


def PG_errorMessage(return_type: str = "traceback", *messages: str) -> str:
    """
    Format error messages with file name mapping and stack traces.

    Equivalent to Translator.pm:533-586

    Process:
    1. Join messages and clean whitespace
    2. Get file mapping from environment
    3. Replace eval IDs with filenames
    4. Add stack trace if requested
    5. Clean up paths (templates → [TMPL], etc.)

    Args:
        return_type: "message" or "traceback"
        *messages: Error messages to format

    Returns:
        Formatted error message with optional stack trace
    """
    message = "\n".join(str(m) for m in messages).rstrip()

    # Get file mapping (try to get from environment)
    files = {}
    tmpl = "$"
    root = "$"
    pg = "$"

    try:
        from .executor import get_environment
        env = get_environment()
        if env:
            files = getattr(env, "_file_mapping", {})
            tmpl = files.get("tmpl", "$")
            root = files.get("root", "$")
            pg = files.get("pg", "$")
    except:
        pass  # No environment available

    # Replace directory paths with abbreviations
    message = message.replace(str(tmpl), "[TMPL]")
    message = message.replace(str(root), "[WW]")
    message = message.replace(str(pg), "[PG]")

    # Replace eval IDs with filenames
    # Find all (eval NNN) or <string> references
    eval_pattern = r"\(eval (\d+)\)|<string>|<stdin>"

    def replace_eval(match: re.Match) -> str:
        """Replace eval reference with filename."""
        eval_ref = match.group(0)
        if eval_ref in files:
            filename = files[eval_ref]
            # Clean up filename
            filename = str(filename).replace(str(tmpl), "[TMPL]")
            filename = filename.replace(str(root), "[WW]")
            filename = filename.replace(str(pg), "[PG]")
            return filename
        return eval_ref

    message = re.sub(eval_pattern, replace_eval, message)

    # Return just message if requested or already has trace
    if return_type == "message" or "Died within" in message or "from within" in message:
        return message + "\n"

    # Remove trailing period for traceback
    message = message.rstrip(".")

    # Build stack trace
    trace_lines = [message]

    # Examine stack frames
    stack = traceback.extract_stack()

    # Skip internal frames
    skip_count = 0
    for i, frame in enumerate(stack):
        if "error_handler.py" in frame.filename:
            skip_count = i + 1
            break

    skip_parser = False

    for i in range(skip_count, len(stack) - 1):  # -1 to skip this function
        frame = stack[i]
        func_name = frame.name

        # Stop at exec or eval
        if func_name in ("exec", "eval", "<module>"):
            continue

        # Skip Parser/Value calls if previous was also Parser/Value
        if skip_parser and ("parser" in frame.filename.lower() or "value" in frame.filename.lower()):
            continue

        skip_parser = "parser" in frame.filename.lower() or "value" in frame.filename.lower()

        # Skip translator internals
        if "pg_translator" in frame.filename and "error_handler" not in frame.filename:
            continue

        # Skip certain PG functions
        if func_name in ("safe_ev", "old_safe_ev", "ev_substring", "<lambda>"):
            continue

        # Get filename
        file = files.get(frame.filename, frame.filename)
        file = str(file).replace(str(tmpl), "[TMPL]")
        file = file.replace(str(root), "[WW]")
        file = file.replace(str(pg), "[PG]")

        # Shorten file path if it's a full path
        if len(file) > 60:
            file_path = Path(file)
            file = f".../{file_path.parent.name}/{file_path.name}"

        trace_line = f"   from within {func_name} called at line {frame.lineno} of {file}"

        # Skip eval references in trace
        if "(eval" not in trace_line and "<string>" not in trace_line:
            trace_lines.append(trace_line)

    return "\n".join(trace_lines) + "\n"


class PGError(Exception):
    """Base exception for PG errors with formatted messages."""

    def __init__(self, message: str, original_exception: Exception | None = None):
        self.original_exception = original_exception
        formatted = PG_errorMessage("traceback", message)
        super().__init__(formatted)


class PGWarning:
    """Warning handler for PG problems."""

    def __init__(self):
        self.frontend_warnings: list[str] = []
        self.backend_warnings: list[str] = []

    def warn(self, message: str, backend: bool = False) -> None:
        """
        Add a warning.

        Args:
            message: Warning message
            backend: If True, only show with debug permission
        """
        if backend:
            self.backend_warnings.append(message)
        else:
            self.frontend_warnings.append(message)

    def get_formatted_warnings(self, has_debug_permission: bool = False) -> str:
        """
        Get formatted warning messages.

        Args:
            has_debug_permission: If True, include backend warnings

        Returns:
            Formatted warnings
        """
        warnings = []

        # Frontend warnings (always show)
        if self.frontend_warnings:
            formatted = PG_errorMessage("message", *self.frontend_warnings)
            warnings.append(formatted)

        # Backend warnings (only with debug permission)
        if self.backend_warnings and has_debug_permission:
            formatted = PG_errorMessage(
                "message",
                "Non-fatal warnings (debugging only):",
                *self.backend_warnings
            )
            warnings.append(formatted)

        return "\n".join(warnings)

    def clear(self) -> None:
        """Clear all warnings."""
        self.frontend_warnings.clear()
        self.backend_warnings.clear()


def install_error_handlers(environment: PGEnvironment) -> tuple:
    """
    Install warning and error handlers for problem execution.

    Args:
        environment: PG environment

    Returns:
        (warning_handler, error_handler) for restoration
    """
    # Create warning tracker
    warning_tracker = PGWarning()

    # Define handlers
    def warning_handler(msg: str, backend: bool = False) -> None:
        warning_tracker.warn(msg, backend)

    def error_handler(msg: str) -> None:
        raise PGError(msg)

    # Store in environment
    environment._warning_tracker = warning_tracker  # type: ignore
    environment._warning_handler = warning_handler  # type: ignore
    environment._error_handler = error_handler  # type: ignore

    return (warning_handler, error_handler)


def format_execution_error(
    error: Exception,
    pg_source: str,
    environment: PGEnvironment | None = None
) -> str:
    """
    Format execution error with problem source context.

    Args:
        error: Exception that occurred
        pg_source: Original PG source code
        environment: PG environment (optional)

    Returns:
        Formatted error message with source context
    """
    # Get formatted error message
    error_msg = PG_errorMessage("traceback", str(error))

    # Add source context if available
    if pg_source and environment:
        has_debug = getattr(environment, "view_problem_debugging_info", False)

        if has_debug:
            # Add source listing with line numbers
            lines = pg_source.split("\n")
            source_listing = []

            for i, line in enumerate(lines, start=1):
                # Escape HTML special chars
                line = line.replace("&", "&amp;")
                line = line.replace("<", "&lt;")
                line = line.replace(">", "&gt;")
                line = line.replace('"', "&quot;")
                line = line.replace("'", "&#39;")

                source_listing.append(f"{i:4d}:\t{line}")

            source_html = "\n".join(source_listing)

            error_msg += (
                "\n<hr>\n"
                "<p>Input Read:</p>\n"
                '<pre style="tab-size:4">\n'
                f"{source_html}\n"
                "</pre>\n"
            )

    return error_msg
