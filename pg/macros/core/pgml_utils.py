"""
PGML Utilities for WeBWorK.

This module provides utilities for generating HTML tags and help links
within PGML (PG Markup Language) contexts.

Based on PGML.pl from the Perl WeBWorK distribution.
"""

from typing import Any, Optional


def tag(tagname: str, content: str = '', **attrs: Any) -> str:
    """
    Generate an HTML tag with content and attributes.
    
    Creates HTML tags with proper attribute formatting and optional content.
    
    Args:
        tagname: Name of the HTML tag (e.g., "div", "span", "p")
        content: Content to place inside the tag (default: empty string)
        **attrs: HTML attributes as keyword arguments
        
    Returns:
        HTML tag string
        
    Example:
        >>> from pg.macros.core.pgml_utils import tag
        >>> tag("div", "Hello World", id="main", class_="container")
        '<div id="main" class_="container">Hello World</div>'
        >>> tag("br", id="break")
        '<br id="break"/>'
    
    Perl Source: PGML.pl tag function
    """
    attr_str = ' '.join(f'{k}="{v}"' for k, v in attrs.items())
    if content:
        return f'<{tagname} {attr_str}>{content}</{tagname}>'
    else:
        return f'<{tagname} {attr_str}/>'


def helpLink(topic: Optional[str] = None, **kwargs: Any) -> str:
    """
    Generate a help documentation link.
    
    Creates a hyperlink to help documentation for a given topic.
    
    Args:
        topic: Help topic identifier
        **kwargs: Additional options (title, target, etc.)
        
    Returns:
        HTML anchor tag string for help link
        
    Example:
        >>> from pg.macros.core.pgml_utils import helpLink
        >>> helpLink("functions")
        '<a href="/help/functions" target="_blank">Help</a>'
    
    Perl Source: PGML.pl helpLink function
    """
    if topic:
        return f'<a href="/help/{topic}" target="_blank">Help</a>'
    else:
        return '<a href="/help" target="_blank">Help</a>'


__all__ = [
    'tag',
    'helpLink',
]

