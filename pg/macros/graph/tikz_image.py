"""TikZ Image Support for WeBWorK.

This module provides the createTikZImage() function for creating images
using TikZ/LaTeX code within WeBWorK problems.

Based on macros/graph/PGtikz.pl from the WeBWorK distribution.
"""

from typing import Any, Dict, List, Optional, Union


class TikZImage:
    """
    TikZ image object for LaTeX-based graphics.

    Creates images from TikZ code that are rendered as PNG, SVG, PDF, or GIF.
    TikZ is a powerful vector graphics language built on LaTeX.

    Attributes:
        tex_code: The TikZ/LaTeX code defining the image
        tikz_options: Options passed to tikzpicture environment
        tikz_libraries: Additional TikZ libraries to load
        tex_packages: LaTeX packages to include
        preamble: Additional TeX preamble commands
        extension: Output format (svg, png, pdf, gif)
        image_name: Filename for generated image
    """

    def __init__(self):
        """Initialize a TikZ image object."""
        self.tex_code = ""
        self.tikz_options = ""
        self.tikz_libraries = []
        self.tex_packages = []
        self.preamble = ""
        self.extension = "svg"  # Default to SVG
        self.image_name = None
        self.svg_method = "pdf2svg"
        self.convert_options = {"input": {}, "output": {}}

    def tex(self, code: str) -> 'TikZImage':
        """
        Set the TikZ code that defines this image.

        Use a single-quoted string to avoid Perl interpolation issues
        with backslashes.

        Args:
            code: TikZ code as a string

        Returns:
            Self for method chaining

        Example:
            >>> image = createTikZImage()
            >>> image.tex(r'''
            ...     \\draw (0,0) -- (2,2);
            ...     \\draw (0,2) -- (2,0);
            ... ''')
        """
        self.tex_code = code
        return self

    def BEGIN_TIKZ(self, code: str) -> 'TikZImage':
        """
        Heredoc-style method for setting TikZ code (Perl compatibility).

        This is provided for compatibility with Perl's heredoc syntax.
        In Python, use tex() method instead.

        Args:
            code: TikZ code

        Returns:
            Self for method chaining
        """
        return self.tex(code)

    def tikzOptions(self, options: str) -> 'TikZImage':
        """
        Set options for the tikzpicture environment.

        Options are passed directly to \\begin{tikzpicture}[options].

        Args:
            options: TikZ options string
                Example: "x=.5cm,y=.5cm,declare function={f(\\x)=sqrt(\\x);}"

        Returns:
            Self for method chaining
        """
        self.tikz_options = options
        return self

    def tikzLibraries(self, libraries: str) -> 'TikZImage':
        """
        Add TikZ libraries to load.

        Multiple libraries can be specified separated by commas.

        Args:
            libraries: Comma-separated library names
                Example: "arrows.meta,calc,shapes.geometric"

        Returns:
            Self for method chaining

        Perl Source: PGtikz.pl documentation
        """
        self.tikz_libraries = [lib.strip() for lib in libraries.split(',')]
        return self

    def texPackages(self, packages: List[Union[str, List[str]]]) -> 'TikZImage':
        """
        Add LaTeX packages to load.

        Args:
            packages: List of package names or [name, options] pairs
                Example: ["pgfplots", ["hf-tikz", "customcolors"], ["xcolor", "cmyk,table"]]

        Returns:
            Self for method chaining

        Perl Source: PGtikz.pl documentation
        """
        self.tex_packages = packages
        return self

    def addToPreamble(self, preamble: str) -> 'TikZImage':
        """
        Add additional commands to the TeX preamble.

        These commands are added after package loading but before the image.

        Args:
            preamble: TeX commands to add to preamble

        Returns:
            Self for method chaining
        """
        self.preamble = preamble
        return self

    def ext(self, extension: str) -> 'TikZImage':
        """
        Set the output image format.

        Valid formats: 'svg' (default), 'png', 'pdf', 'gif'

        In hardcopy/TeX mode, this is automatically set to 'pdf'.

        Args:
            extension: File extension (without dot)

        Returns:
            Self for method chaining

        Perl Source: PGtikz.pl lines 115-120
        """
        # In TeX/hardcopy mode, pdf is used; in PTX mode, tgz is used
        if extension not in ('svg', 'png', 'pdf', 'gif', 'tgz'):
            raise ValueError(f"Invalid extension '{extension}'. Must be svg, png, pdf, or gif.")
        self.extension = extension
        return self

    def svgMethod(self, method: str) -> 'TikZImage':
        """
        Set the method for converting PDF to SVG.

        Args:
            method: Conversion method (typically 'pdf2svg')

        Returns:
            Self for method chaining
        """
        self.svg_method = method
        return self

    def convertOptions(self, options: Dict[str, Any]) -> 'TikZImage':
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

        Perl Source: PGtikz.pl documentation
        """
        self.convert_options = options
        return self

    def imageName(self, name: str) -> 'TikZImage':
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
        return f"TikZImage(ext={self.extension}, code_len={len(self.tex_code)})"


def createTikZImage() -> TikZImage:
    """
    Create a TikZ image object for use in WeBWorK problems.

    TikZ is a powerful vector graphics language that allows precise control
    over graphics. The generated images are rendered as SVG, PNG, PDF, or GIF.

    Returns:
        New TikZImage object configured for SVG output by default

    Example:
        >>> image = createTikZImage()
        >>> image.tex(r'''
        ...     \\begin{tikzpicture}
        ...     \\draw (-2,0) -- (2,0);
        ...     \\draw (0,-2) -- (0,2);
        ...     \\draw (0,0) circle[radius=1.5];
        ...     \\end{tikzpicture}
        ... ''')
        >>> image.tikzLibraries("arrows.meta,calc")
        >>> image.ext("png")

    Perl Source: macros/graph/PGtikz.pl lines 91-93, 100-113
    """
    return TikZImage()


__all__ = [
    'TikZImage',
    'createTikZImage',
]
