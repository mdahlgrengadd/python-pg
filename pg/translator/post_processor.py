"""
Content Post-Processing System for PG Translator.

Manages hooks for modifying problem content after rendering.
Reference: Translator.pm:1165-1207
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ContentPostProcessor:
    """
    Manages content post-processing hooks.

    Post-processors can modify problem text, header text, and add
    interactive features after rendering is complete.
    """

    def __init__(self):
        """Initialize post-processor."""
        self.processors: list[Callable] = []

    def add_processor(self, processor: Callable) -> None:
        """
        Add a post-processor hook.

        Args:
            processor: Function that takes (problem_dom, header_dom, problem_result)
                      or (problem_text_ref) for TeX mode
        """
        self.processors.append(processor)

    def process(
        self,
        problem_text: str,
        header_text: str,
        display_mode: str,
        problem_result: dict[str, Any] | None = None
    ) -> tuple[str, str]:
        """
        Run all post-processors.

        Args:
            problem_text: Problem HTML/TeX
            header_text: Header HTML
            display_mode: "HTML", "TeX", or "PTX"
            problem_result: Grading result (optional)

        Returns:
            (processed_problem_text, processed_header_text)
        """
        if not self.processors:
            # No processors, return as-is
            return (problem_text, header_text)

        if display_mode == "TeX":
            # TeX mode: pass text reference to processors
            text_ref = {"text": problem_text}

            for processor in self.processors:
                try:
                    processor(text_ref)
                except Exception as e:
                    logger.error(f"Post-processor error (TeX mode): {e}", exc_info=True)

            return (text_ref["text"], header_text)

        else:
            # HTML/PTX: use DOM manipulation
            try:
                from lxml import html as lxml_html
                from lxml import etree

                # Parse HTML
                # Use fragment parser to avoid adding html/body tags
                problem_dom = lxml_html.fragment_fromstring(
                    problem_text,
                    create_parent="div"
                )

                header_dom = lxml_html.fragment_fromstring(
                    header_text or "<div></div>",
                    create_parent="div"
                )

                # Run processors
                for processor in self.processors:
                    try:
                        processor(problem_dom, header_dom, problem_result or {})
                    except Exception as e:
                        logger.error(f"Post-processor error (HTML mode): {e}", exc_info=True)

                # Convert back to strings
                problem_output = etree.tostring(
                    problem_dom,
                    encoding="unicode",
                    method="html"
                )

                header_output = etree.tostring(
                    header_dom,
                    encoding="unicode",
                    method="html"
                )

                # Remove wrapper div if we added it
                if problem_output.startswith("<div>") and problem_output.endswith("</div>"):
                    problem_output = problem_output[5:-6]

                if header_output.startswith("<div>") and header_output.endswith("</div>"):
                    header_output = header_output[5:-6]

                return (problem_output, header_output)

            except ImportError:
                # lxml not available, return as-is
                logger.warning("lxml not available, skipping DOM post-processing")
                return (problem_text, header_text)
            except Exception as e:
                logger.error(f"Error in post-processing: {e}", exc_info=True)
                # Return original on error
                return (problem_text, header_text)


def add_content_post_processor(processor: Callable) -> None:
    """
    Add a content post-processor hook (for use in PG problems).

    Usage in PG problems:
        add_content_post_processor(lambda dom, header, result: ...)

    Args:
        processor: Post-processor function
    """
    from .executor import get_environment

    env = get_environment()
    if env is None:
        raise RuntimeError("No active PG environment")

    if not hasattr(env, "post_processors"):
        env.post_processors = ContentPostProcessor()  # type: ignore

    env.post_processors.add_processor(processor)  # type: ignore


# Example post-processors for common use cases

def add_warning_style_processor(problem_dom, header_dom, result):
    """
    Add warning styling to problems with incorrect answers.

    This is an example post-processor that can be registered.
    """
    try:
        from lxml import html as lxml_html

        score = result.get("score", 1)

        if score < 1:
            # Add warning class to problem
            existing_class = problem_dom.get("class", "")
            problem_dom.set("class", existing_class + " has-incorrect-answers")

            # Add CSS to header
            style = lxml_html.Element("style")
            style.text = """
            .has-incorrect-answers {
                border-left: 4px solid #ff9800;
                padding-left: 1em;
                background-color: #fff3e0;
            }
            """
            header_dom.append(style)

    except Exception:
        pass  # Silently fail


def add_accessibility_processor(problem_dom, header_dom, result):
    """
    Add accessibility attributes to problem elements.

    Example post-processor for ARIA labels and semantic HTML.
    """
    try:
        # Add role="math" to math elements
        for elem in problem_dom.xpath(".//*[contains(@class, 'math')]"):
            elem.set("role", "math")

        # Add aria-label to answer inputs
        for input_elem in problem_dom.xpath(".//input[@type='text']"):
            name = input_elem.get("name", "")
            if name and not input_elem.get("aria-label"):
                input_elem.set("aria-label", f"Answer for {name}")

    except Exception:
        pass  # Silently fail


# Registry of built-in post-processors
BUILTIN_PROCESSORS = {
    "warning_style": add_warning_style_processor,
    "accessibility": add_accessibility_processor,
}


def get_builtin_processor(name: str) -> Callable:
    """
    Get a built-in post-processor by name.

    Args:
        name: Processor name

    Returns:
        Post-processor function

    Raises:
        KeyError: If processor not found
    """
    return BUILTIN_PROCESSORS[name]
