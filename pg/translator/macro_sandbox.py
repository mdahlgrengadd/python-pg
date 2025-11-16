"""
Enhanced Sandbox with Macro Support.

Extends the subprocess sandbox to support loading and executing PG macros.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .sandbox import Sandbox, SandboxResult


class MacroEnabledSandbox(Sandbox):
    """
    Sandbox with macro loading support.

    Enhances the subprocess sandbox to:
    - Load PG macro modules (pg_core, pg_basic_macros)
    - Inject macro functions into problem namespace
    - Support PGEnvironment integration
    """

    def __init__(self, timeout: int = 30):
        """
        Initialize macro-enabled sandbox.

        Args:
            timeout: Maximum execution time in seconds
        """
        super().__init__(timeout=timeout)
        self.macro_modules: list[str] = []

    def load_macros(self, *macro_names: str) -> None:
        """
        Mark macros to be loaded in sandbox.

        Args:
            *macro_names: Macro names (e.g., "PG.pl", "PGbasicmacros.pl")
        """
        for name in macro_names:
            # Convert Perl names to Python module names
            py_name = self._perl_to_python_name(name)
            if py_name not in self.macro_modules:
                self.macro_modules.append(py_name)

    def _perl_to_python_name(self, perl_name: str) -> str:
        """
        Convert Perl macro name to Python module name.

        Args:
            perl_name: Perl macro name (e.g., "PG.pl", "PGbasicmacros.pl")

        Returns:
            Python module name (e.g., "pg_core", "pg_basic_macros")
        """
        # Remove .pl extension
        name = perl_name.removesuffix(".pl")

        # Convert to Python naming
        mapping = {
            "PG": "pg_core",
            "PGbasicmacros": "pg_basic_macros",
            "PGstandard": "pg_standard",
            "PGML": "pgml",
            "PGanswermacros": "pg_answer_macros",
        }

        return mapping.get(name, name.lower().replace("pg", "pg_"))

    def execute_with_macros(
        self,
        code: str,
        seed: int,
        context: Any = None,
        globals_dict: dict[str, Any] | None = None
    ) -> SandboxResult:
        """
        Execute code with macro support.

        Args:
            code: Python code to execute
            seed: Random seed
            context: Mathematical context
            globals_dict: Additional global variables

        Returns:
            SandboxResult with execution results
        """
        # Build macro imports
        macro_imports = self._build_macro_imports()

        # Wrap code with macro setup
        wrapped_code = self._wrap_with_macros(code, macro_imports)

        # Execute
        return self.execute(wrapped_code, seed, context, globals_dict)

    def _build_macro_imports(self) -> str:
        """
        Build import statements for loaded macros.

        Returns:
            Python import code
        """
        if not self.macro_modules:
            return ""

        imports = []
        for module in self.macro_modules:
            if module == "pg_core":
                imports.append("""
# Import PG core macros
try:
    from pg.macros.core.pg_core import (
        DOCUMENT, ENDDOCUMENT,
        TEXT, BEGIN_TEXT, END_TEXT,
        ANS, NAMED_ANS, NEW_ANS_NAME,
        SOLUTION, HINT, COMMENT,
        random, non_zero_random, list_random,
        loadMacros,
        PGEnvironment, set_environment, get_environment
    )
except ImportError:
    # Fallback: define stub functions
    def DOCUMENT(): pass
    def ENDDOCUMENT(): pass
    def TEXT(*args): pg_env.add_text(' '.join(str(a) for a in args))
    def BEGIN_TEXT(): return ''
    def END_TEXT(): return ''
    def ANS(*args):
        for ev in args:
            pg_env.register_answer(f'AnSwEr{len(pg_env.answers):04d}', ev)
    def NAMED_ANS(name, ev): pg_env.register_answer(name, ev)
    def NEW_ANS_NAME(): return f'AnSwEr{len(pg_env.answers):04d}'
    def SOLUTION(*args): pg_env.add_solution(' '.join(str(a) for a in args))
    def HINT(*args): pg_env.add_hint(' '.join(str(a) for a in args))
    def COMMENT(*args): pass
    def random(a=0, b=1, step=None):
        import random as _r
        if step:
            return a + _r.randint(0, int((b-a)/step)) * step
        return a + _r.random() * (b - a)
    def non_zero_random(a, b, step=None):
        while True:
            v = random(a, b, step)
            if v != 0: return v
    def list_random(*items):
        import random as _r
        return _r.choice(items)
    def loadMacros(*args): pass
    def set_environment(env): pass
    def get_environment(): return pg_env
    class PGEnvironment:
        def __init__(self, envir):
            self.output_array = []
            self.answers_hash = {}
""")

            elif module == "pg_basic_macros":
                imports.append("""
# Import PG basic macros
try:
    from pg.macros.core.pg_basic_macros import (
        ans_rule, ans_box, ans_radio_buttons, pop_up_list,
        NAMED_ANS_RULE, NAMED_ANS_BOX, NAMED_ANS_RADIO_BUTTONS, NAMED_POP_UP_LIST,
        PAR, BR, BBOLD, EBOLD, BITALIC, EITALIC,
        BCENTER, ECENTER, BUL, EUL,
        MODES, image, PI, E
    )
except ImportError:
    # Fallback: define stub functions
    def ans_rule(width=20):
        name = NEW_ANS_NAME()
        return f'<input type="text" name="{name}" size="{width}"/>'
    def ans_box(rows=5, cols=20):
        name = NEW_ANS_NAME()
        return f'<textarea name="{name}" rows="{rows}" cols="{cols}"></textarea>'
    def ans_radio_buttons(choices, **opts):
        name = NEW_ANS_NAME()
        html = []
        for i, (val, label) in enumerate(choices):
            html.append(f'<label><input type="radio" name="{name}" value="{val}"/> {label}</label>')
        return opts.get('separator', '<br/>').join(html)
    def pop_up_list(choices, **opts):
        name = NEW_ANS_NAME()
        if isinstance(choices, dict):
            opts_html = [f'<option value="{k}">{v}</option>' for k, v in choices.items()]
        else:
            opts_html = [f'<option value="{c}">{c}</option>' for c in choices]
        return f'<select name="{name}">{\"\".join(opts_html)}</select>'
    def NAMED_ANS_RULE(name, width=20):
        return f'<input type="text" name="{name}" size="{width}"/>'
    def NAMED_ANS_BOX(name, rows=5, cols=20):
        return f'<textarea name="{name}" rows="{rows}" cols="{cols}"></textarea>'
    def NAMED_ANS_RADIO_BUTTONS(name, choices, **opts):
        html = []
        for i, (val, label) in enumerate(choices):
            html.append(f'<label><input type="radio" name="{name}" value="{val}"/> {label}</label>')
        return opts.get('separator', '<br/>').join(html)
    def NAMED_POP_UP_LIST(name, choices, **opts):
        if isinstance(choices, dict):
            opts_html = [f'<option value="{k}">{v}</option>' for k, v in choices.items()]
        else:
            opts_html = [f'<option value="{c}">{c}</option>' for c in choices]
        return f'<select name="{name}">{\"\".join(opts_html)}</select>'
    def PAR(): return '<p>'
    def BR(): return '<br/>'
    def BBOLD(): return '<strong>'
    def EBOLD(): return '</strong>'
    def BITALIC(): return '<em>'
    def EITALIC(): return '</em>'
    def BCENTER(): return '<div style="text-align: center;">'
    def ECENTER(): return '</div>'
    def BUL(): return '<u>'
    def EUL(): return '</u>'
    def MODES(**kwargs): return kwargs.get('HTML', '')
    def image(filename, **opts): return f'<img src="{filename}"/>'
    import math
    def PI(): return math.pi
    def E(): return math.e
""")

        return "\n".join(imports)

    def _wrap_with_macros(self, code: str, macro_imports: str) -> str:
        """
        Wrap code with macro imports.

        Args:
            code: User code
            macro_imports: Macro import statements

        Returns:
            Wrapped code
        """
        return f"""
{macro_imports}

# User code begins here
{code}
"""

    def _wrap_code(
        self, code: str, seed: int, context: Any = None, globals_dict: dict[str, Any] | None = None
    ) -> str:
        """
        Override parent _wrap_code to include macro support.

        Args:
            code: User's PG code
            seed: Random seed
            context: Mathematical context
            globals_dict: Additional globals

        Returns:
            Complete Python script
        """
        # Get base wrapped code from parent
        base_code = super()._wrap_code(code, seed, context, globals_dict)

        # Build macro imports
        macro_imports = self._build_macro_imports()

        # Insert macro imports after the PGEnv class definition
        # Find the line "pg_env = PGEnv()"
        lines = base_code.split('\n')
        insert_index = 0
        for i, line in enumerate(lines):
            if 'pg_env = PGEnv()' in line:
                insert_index = i + 1
                break

        # Insert macro imports
        if insert_index > 0 and macro_imports:
            lines.insert(insert_index, "\n" + macro_imports + "\n")

        return '\n'.join(lines)


def create_macro_sandbox(timeout: int = 30) -> MacroEnabledSandbox:
    """
    Create a macro-enabled sandbox.

    Args:
        timeout: Maximum execution time

    Returns:
        MacroEnabledSandbox instance
    """
    return MacroEnabledSandbox(timeout=timeout)
