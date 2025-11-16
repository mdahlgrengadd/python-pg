"""LaTeX Image Support for WeBWorK.

This module provides the createLaTeXImage() function for creating images
using LaTeX code within WeBWorK problems.

Based on macros/graph/PGlateximage.pl from the WeBWorK distribution.
"""

from typing import Any, Dict, List, Optional, Union


class LaTeXImage:
    """
    LaTeX image object for rendering LaTeX-based graphics.

    Creates images from LaTeX/TeX code that are rendered as PNG, SVG, PDF, or GIF.
    Supports custom environments, packages, and preamble additions.

    Attributes:
        tex_code: The LaTeX/TeX code defining the image
        environment: The LaTeX environment to wrap code in
        environment_options: Options for the environment
        tex_packages: LaTeX packages to include
        tikz_libraries: TikZ libraries to load (if using tikz environment)
        preamble: Additional TeX preamble commands
        extension: Output format (svg, png, pdf, gif)
        image_name: Filename for generated image
    """

    def __init__(self):
        """Initialize a LaTeX image object."""
        self.tex_code = ""
        self._environment = None
        self.environment_options = ""
        self.tex_packages = []
        self.tikz_libraries = []
        self.preamble = ""
        self.extension = "svg"  # Default to SVG
        self.image_name = None
        self.svg_method = "pdf2svg"
        self.convert_options = {"input": {}, "output": {}}

    def tex(self, code: str) -> 'LaTeXImage':
        """
        Set the LaTeX code that defines this image.

        Use a single-quoted string to avoid Perl interpolation issues
        with backslashes.

        Args:
            code: LaTeX code as a string

        Returns:
            Self for method chaining

        Example:
            >>> image = createLaTeXImage()
            >>> image.tex(r'''
            ...     \\begin{tabular}{cc}
            ...     a & b \\\\
            ...     c & d
            ...     \\end{tabular}
            ... ''')
        """
        self.tex_code = code
        return self

    def BEGIN_LATEX_IMAGE(self, code: str) -> 'LaTeXImage':
        """
        Heredoc-style method for setting LaTeX code (Perl compatibility).

        This is provided for compatibility with Perl's heredoc syntax.
        In Python, use tex() method instead.

        Args:
            code: LaTeX code

        Returns:
            Self for method chaining
        """
        return self.tex(code)

    def environment(self, env: Union[str, List[str]]) -> 'LaTeXImage':
        """
        Set the LaTeX environment to wrap the code in.

        Common environments: 'tikzpicture', 'tabular', 'align', 'circuitikz', etc.

        Args:
            env: Either a string environment name or [name, options] pair
                Example: "tikzpicture" or ["circuitikz", "scale=1.2, transform shape"]

        Returns:
            Self for method chaining

        Perl Source: PGlateximage.pl documentation
        """
        if isinstance(env, list):
            if len(env) > 0:
                self._environment = env[0]
            if len(env) > 1:
                self.environment_options = env[1]
        else:
            self._environment = env
        return self

    def texPackages(self, packages: List[Union[str, List[str]]]) -> 'LaTeXImage':
        """
        Add LaTeX packages to load.

        Args:
            packages: List of package names or [name, options] pairs
                Example: ["pgfplots", ["hf-tikz", "customcolors"], ["xcolor", "cmyk,table"]]

        Returns:
            Self for method chaining

        Perl Source: PGlateximage.pl documentation
        """
        self.tex_packages = packages
        return self

    def tikzLibraries(self, libraries: str) -> 'LaTeXImage':
        """
        Add TikZ libraries to load.

        Multiple libraries can be specified separated by commas.
        Only relevant if using tikzpicture environment.

        Args:
            libraries: Comma-separated library names
                Example: "arrows.meta,calc,shapes.geometric"

        Returns:
            Self for method chaining

        Perl Source: PGlateximage.pl documentation
        """
        self.tikz_libraries = [lib.strip() for lib in libraries.split(',')]
        return self

    def addToPreamble(self, preamble: str) -> 'LaTeXImage':
        """
        Add additional commands to the TeX preamble.

        These commands are added after package loading but before the image.

        Args:
            preamble: TeX commands to add to preamble

        Returns:
            Self for method chaining

        Perl Source: PGlateximage.pl documentation
        """
        self.preamble = preamble
        return self

    def ext(self, extension: str) -> 'LaTeXImage':
        """
        Set the output image format.

        Valid formats: 'svg' (default), 'png', 'pdf', 'gif'

        In hardcopy/TeX mode, this is automatically set to 'pdf'.

        Args:
            extension: File extension (without dot)

        Returns:
            Self for method chaining

        Perl Source: PGlateximage.pl lines 115-120
        """
        if extension not in ('svg', 'png', 'pdf', 'gif', 'tgz'):
            raise ValueError(f"Invalid extension '{extension}'. Must be svg, png, pdf, or gif.")
        self.extension = extension
        return self

    def svgMethod(self, method: str) -> 'LaTeXImage':
        """
        Set the method for converting PDF to SVG.

        Args:
            method: Conversion method (typically 'pdf2svg')

        Returns:
            Self for method chaining
        """
        self.svg_method = method
        return self

    def convertOptions(self, options: Dict[str, Any]) -> 'LaTeXImage':
        """
        Set ImageMagick convert options for PNG output.

        These options are used when converting the image using ImageMagick.

        Args:
            options: Dict with 'input' and 'output' keys containing
                ImageMagick conversion options
                Example: {"input": {"density": 300},
                         "output": {"quality": 100, "resize": "500x500"}}

        Returns:
            Self for method chaining

        Perl Source: PGlateximage.pl documentation
        """
        self.convert_options = options
        return self

    def imageName(self, name: str) -> 'LaTeXImage':
        """
        Set the filename for the generated image.

        Args:
            name: Image filename (usually auto-generated)

        Returns:
            Self for method chaining
        """
        self.image_name = name
        return self

    def __repr__(self) -> str:
        """Return string representation."""
        env_str = f", env={self._environment}" if self._environment else ""
        return f"LaTeXImage(ext={self.extension}{env_str}, code_len={len(self.tex_code)})"


def createLaTeXImage() -> LaTeXImage:
    """
    Create a LaTeX image object for use in WeBWorK problems.

    Supports custom LaTeX code with configurable environments, packages, and options.
    Images can be rendered as SVG, PNG, PDF, or GIF.

    Returns:
        New LaTeXImage object configured for SVG output by default

    Example:
        >>> image = createLaTeXImage()
        >>> image.texPackages([['xy', 'all']])
        >>> image.tex(r'''
        ...     \\xymatrix{
        ...         A \\ar[r] & B \\ar[d] \\\\
        ...         D \\ar[u] & C \\ar[l]
        ...     }
        ... ''')
        >>> image.ext("png")

    Perl Source: macros/graph/PGlateximage.pl lines 92-93, 101-113
    """
    return LaTeXImage()


__all__ = [
    'LaTeXImage',
    'createLaTeXImage',
]
