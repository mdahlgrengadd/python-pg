"""
IR (Intermediate Representation) emission to Python code.

This module converts the IR tuples produced by the AST transformer
into Python code strings.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional


class PGIREmitter:
    """
    Emits Python code from intermediate representation (IR).

    The IR is a tree of tuples representing Perl constructs. This class
    traverses the IR and generates equivalent Python code with proper
    indentation and syntax.
    """

    def __init__(self, pygments_rewriter):
        """
        Initialize the IR emitter.

        Args:
            pygments_rewriter: PGPygmentsRewriter instance for fallback rewriting
        """
        self._pygments_rewriter = pygments_rewriter
        self._closure_counter = 0

    def desigil(self, name: str) -> str:
        """Remove Perl sigils from variable names."""
        if name and name[0] in '$@%':
            return name[1:]
        return name

    def emit_ir(self, ir: Any, indent: int = 0) -> Optional[str]:
        """
        Convert IR nodes into Python code lines.

        Args:
            ir: IR tuple or primitive value
            indent: Current indentation level

        Returns:
            Python code string or None for no output
        """
        if not isinstance(ir, tuple):
            return None

        typ = ir[0]
        ind = "    " * indent

        if typ == "noop":
            return None

        # Control flow statements
        if typ == "if":
            return self._emit_if(ir, indent)

        if typ == "unless":
            return self._emit_unless(ir, indent)

        if typ == "while":
            return self._emit_while(ir, indent)

        if typ in {"for", "foreach"}:
            return self._emit_for(ir, indent)

        if typ == "do_until":
            return self._emit_do_until(ir, indent)

        if typ == "stmt_modifier":
            return self._emit_stmt_modifier(ir, indent)

        # Assignments
        if typ == "assign":
            return self._emit_assign(ir, indent)

        if typ == "subscript_assign":
            return self._emit_subscript_assign(ir, indent)

        # Function calls
        if typ == "call":
            _, name, args = ir
            if name == "loadMacros":
                return None
            py_args = [self.expr_to_py(a) for a in args]
            return f"{ind}{name}({', '.join(py_args)})"

        # Expressions
        if typ == "expr":
            _, expr = ir
            return f"{ind}{self.expr_to_py(expr)}"

        if typ == "bin":
            return f"{ind}{self.expr_to_py(ir)}"

        # Closures
        if typ == "closure":
            _, body_stmts = ir
            return self.emit_closure(body_stmts, indent)

        # Return statement
        if typ == "return":
            _, return_value = ir
            if return_value is None or (isinstance(return_value, str) and return_value == ""):
                return f"{ind}return"
            return_py = self.expr_to_py(return_value)
            return f"{ind}return {return_py}"

        # Parameter unpacking
        if typ == "param_unpack":
            _, var_list, deref_source = ir
            var_names = []
            for var_tuple in var_list:
                var_name = self.desigil(var_tuple[1] if isinstance(var_tuple, tuple) else var_tuple)
                var_names.append(var_name)
            vars_str = ", ".join(var_names)
            deref_py = self.expr_to_py(deref_source)
            return f"{ind}{vars_str} = {deref_py}"

        # Unknown IR: produce raw comment
        return f"{ind}# {ir}"

    def _emit_if(self, ir: tuple, indent: int) -> str:
        """Emit if/elsif/else statement."""
        _, condition, block, clauses = ir
        cond_py = self.expr_to_py(condition)
        block_stmts = self.emit_block(block, indent + 1)

        lines = [f"{'    ' * indent}if {cond_py}:"]
        lines.extend(block_stmts)

        for clause in clauses:
            if clause[0] == "elsif":
                _, elif_cond, elif_block = clause
                elif_cond_py = self.expr_to_py(elif_cond)
                lines.append(f"{'    ' * indent}elif {elif_cond_py}:")
                lines.extend(self.emit_block(elif_block, indent + 1))
            elif clause[0] == "else":
                _, else_block = clause
                lines.append(f"{'    ' * indent}else:")
                lines.extend(self.emit_block(else_block, indent + 1))

        return "\n".join(lines)

    def _emit_unless(self, ir: tuple, indent: int) -> str:
        """Emit unless statement (if not)."""
        _, condition, block = ir
        cond_py = self.expr_to_py(condition)
        block_stmts = self.emit_block(block, indent + 1)
        lines = [f"{'    ' * indent}if not ({cond_py}):"]
        lines.extend(block_stmts)
        return "\n".join(lines)

    def _emit_while(self, ir: tuple, indent: int) -> str:
        """Emit while loop."""
        _, condition, block = ir
        cond_py = self.expr_to_py(condition)
        block_stmts = self.emit_block(block, indent + 1)
        lines = [f"{'    ' * indent}while {cond_py}:"]
        lines.extend(block_stmts)
        return "\n".join(lines)

    def _emit_for(self, ir: tuple, indent: int) -> str:
        """Emit for/foreach loop."""
        _, var, expr, block = ir
        var_name = self.desigil(var[1] if isinstance(var, tuple) else var)
        expr_py = self.expr_to_py(expr)
        block_stmts = self.emit_block(block, indent + 1)
        lines = [f"{'    ' * indent}for {var_name} in {expr_py}:"]
        lines.extend(block_stmts)
        return "\n".join(lines)

    def _emit_do_until(self, ir: tuple, indent: int) -> str:
        """Emit do-until loop."""
        _, block, condition = ir
        cond_py = self.expr_to_py(condition)
        block_stmts = self.emit_block(block, indent + 1)
        ind = "    " * indent
        lines = [f"{ind}while True:"]
        lines.extend(block_stmts)
        lines.append(f"{ind}    if ({cond_py}):")
        lines.append(f"{ind}        break")
        return "\n".join(lines)

    def _emit_stmt_modifier(self, ir: tuple, indent: int) -> str:
        """Emit statement modifier (if/unless)."""
        _, stmt, modifier, condition = ir
        stmt_py = self.emit_ir(stmt, indent)
        cond_py = self.expr_to_py(condition)
        if cond_py.startswith('(') and cond_py.endswith(')'):
            cond_py = cond_py[1:-1]

        ind = "    " * indent
        if modifier == "if":
            return f"{ind}if ({cond_py}): {stmt_py.strip()}"
        else:  # unless
            return f"{ind}if not ({cond_py}): {stmt_py.strip()}"

    def _emit_assign(self, ir: tuple, indent: int) -> str:
        """Emit variable assignment."""
        _, var, expr = ir
        var_name = self.desigil(var[1] if isinstance(var, tuple) else var)
        expr_py = self.expr_to_py(expr)
        if expr_py.startswith('(') and expr_py.endswith(')') and isinstance(expr, tuple) and expr[0] == "bin":
            expr_py = expr_py[1:-1]
        return f"{'    ' * indent}{var_name} = {expr_py}"

    def _emit_subscript_assign(self, ir: tuple, indent: int) -> str:
        """Emit subscript assignment."""
        _, var, subscript, expr = ir
        var_name = self.desigil(var[1] if isinstance(var, tuple) else var)
        subscript_py = self.emit_subscript(subscript)
        expr_py = self.expr_to_py(expr)
        return f"{'    ' * indent}{var_name}{subscript_py} = {expr_py}"

    def emit_block(self, block_ir: Any, indent: int) -> List[str]:
        """Emit a block of statements with proper indentation."""
        if not isinstance(block_ir, tuple) or block_ir[0] != "block":
            return []

        _, stmts = block_ir
        lines = []
        for stmt in stmts:
            emitted = self.emit_ir(stmt, indent)
            if emitted:
                lines.append(emitted)

        if not lines:
            lines.append("    " * indent + "pass")

        return lines

    def emit_subscript(self, subscript_ir: Any) -> str:
        """Emit a subscript operation (array or hash)."""
        typ = subscript_ir[0]
        _, expr = subscript_ir
        expr_py = self.expr_to_py(expr)

        if typ == "array_subscript":
            return f"[{expr_py}]"
        elif typ == "hash_subscript":
            if expr_py and expr_py[0] not in ('"', "'"):
                return f"['{expr_py}']"
            return f"[{expr_py}]"

        return f"[{expr_py}]"

    def emit_closure(self, body_stmts: List[Any], indent: int, context_name: str = "func") -> Any:
        """
        Emit a Python function for a Perl closure (sub { ... }).

        Returns either a lambda string (for simple closures) or a tuple of
        (function_definition_lines, function_name) for complex closures.

        Args:
            body_stmts: List of IR tuples representing closure body
            indent: Current indentation level
            context_name: Name for function naming

        Returns:
            str: Lambda expression for simple closures
            Tuple[List[str], str]: Function definition + reference for complex
        """
        params = []
        body_for_return = []
        first_param_unpack = True

        for stmt in body_stmts:
            if isinstance(stmt, tuple) and stmt[0] == "param_unpack" and first_param_unpack:
                _, var_list, deref_source = stmt
                for var_tuple in var_list:
                    var_name = self.desigil(var_tuple[1] if isinstance(var_tuple, tuple) else var_tuple)
                    params.append(var_name)
                first_param_unpack = False
            else:
                body_for_return.append(stmt)

        params_str = ", ".join(params) if params else "*args, **kwargs"

        # Check if body is a single return statement
        if (len(body_for_return) == 1 and
            isinstance(body_for_return[0], tuple) and
                body_for_return[0][0] == "return"):
            _, return_value = body_for_return[0]
            return_py = self.expr_to_py(return_value)
            return f"lambda {params_str}: {return_py}"

        # Complex closure - extract to separate def function
        func_name = self._generate_closure_name(context_name)
        func_lines = []
        ind = "    " * indent

        func_lines.append(f"{ind}def {func_name}({params_str}):")

        if body_for_return:
            for stmt in body_for_return:
                emitted = self.emit_ir(stmt, indent + 1)
                if emitted:
                    func_lines.append(emitted)
        else:
            func_lines.append(f"{ind}    pass")

        return (func_lines, func_name)

    def _generate_closure_name(self, context_name: str) -> str:
        """Generate unique function name for extracted closure."""
        self._closure_counter += 1
        counter_hex = format(self._closure_counter, 'x')
        safe_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in context_name)
        safe_name = safe_name.strip('_') or 'closure'
        return f"_closure_{safe_name}_{counter_hex}"

    def expr_to_py(self, expr: Any) -> str:
        """
        Lower an expression IR into a Python expression string.

        Args:
            expr: IR tuple or primitive value

        Returns:
            Python expression string
        """
        if isinstance(expr, tuple):
            head = expr[0]

            if head == "var":
                return self.desigil(expr[1])

            if head == "bin":
                return self._expr_binary(expr)

            if head == "ternary":
                _, cond, true_val, false_val = expr
                cond_py = self.expr_to_py(cond)
                true_py = self.expr_to_py(true_val)
                false_py = self.expr_to_py(false_val)
                return f"({true_py} if {cond_py} else {false_py})"

            if head == "range":
                _, start, end = expr
                start_py = self.expr_to_py(start)
                end_py = self.expr_to_py(end)
                return f"range({start_py}, {end_py} + 1)"

            if head == "unary":
                _, op, operand = expr
                operand_py = self.expr_to_py(operand)
                if op == "!":
                    return f"(not {operand_py})"
                return f"({op}{operand_py})"

            if head == "postfix":
                return self._expr_postfix(expr)

            if head == "call":
                return self._expr_call(expr)

            if head == "map":
                _, block_expr, list_expr = expr
                block_py = self.expr_to_py(block_expr)
                list_py = self.expr_to_py(list_expr)
                return f"[{block_py} for _ in {list_py}]"

            if head == "grep":
                _, block_expr, list_expr = expr
                block_py = self.expr_to_py(block_expr)
                list_py = self.expr_to_py(list_expr)
                return f"[_ for _ in {list_py} if {block_py}]"

            if head == "array_deref":
                _, var = expr
                return self.expr_to_py(var)

            if head == "special_var":
                _, var = expr
                if var == "@_":
                    return "args"
                return var

            if head == "array":
                _, items = expr
                item_strs = [self.expr_to_py(item) for item in items]
                return f"[{', '.join(item_strs)}]"

            if head == "regex":
                _, pattern, flags = expr
                flag_map = {
                    'i': 're.IGNORECASE', 'm': 're.MULTILINE',
                    's': 're.DOTALL', 'x': 're.VERBOSE',
                }
                py_flags = ' | '.join(flag_map.get(f, '') for f in str(flags) if f in flag_map)
                if py_flags:
                    return f're.compile(r"{pattern}", {py_flags})'
                return f're.compile(r"{pattern}")'

            if head == "expr":
                return self.expr_to_py(expr[1])

            if head == "assign":
                _, var, val = expr
                return f"{self.desigil(var[1])} = {self.expr_to_py(val)}"

        # Fallback to Pygments rewriting
        rewritten = self._pygments_rewriter.rewrite_with_pygments(str(expr))
        return rewritten

    def _expr_binary(self, expr: tuple) -> str:
        """Convert binary expression to Python."""
        _, left, op, right = expr
        py_left = self.expr_to_py(left)
        py_right = self.expr_to_py(right)

        op_map = {
            "eq": "==", "ne": "!=", "lt": "<", "gt": ">",
            "le": "<=", "ge": ">=",
            ".": "+", "x": "*",
            "||": " or ", "&&": " and ",
            "or": " or ", "and": " and "
        }
        py_op = op_map.get(op, op)

        if op == ".":
            return f"pg_concat({py_left}, {py_right})"
        elif op == "x":
            return f"pg_repeat({py_left}, {py_right})"

        return f"({py_left} {py_op} {py_right})"

    def _expr_postfix(self, expr: tuple) -> str:
        """Convert postfix expression (method calls, subscripts) to Python."""
        _, base, op = expr
        base_py = self.expr_to_py(base)

        if isinstance(op, tuple):
            if op[0] == "method_call":
                _, method_name, args = op

                if method_name == "reduce" and len(args) == 0:
                    return f"{base_py}.reduce"

                property_names = {
                    "transpose", "inverse", "norm", "dimensions",
                    "trace", "det", "determinant", "value"
                }
                if method_name in property_names and len(args) == 0:
                    return f"{base_py}.{method_name}"

                if method_name == "with":
                    method_name = "with_params"

                arg_strs = []
                for a in args:
                    if isinstance(a, tuple) and len(a) >= 2 and a[0] == "named_param":
                        _, key_expr, val_expr = a
                        key_str = self.expr_to_py(key_expr)
                        val_str = self.expr_to_py(val_expr)
                        if isinstance(key_expr, tuple) and key_expr[0] == "var":
                            key_str = key_expr[1]
                        arg_strs.append(f"{key_str} = {val_str}")
                    else:
                        arg_strs.append(self.expr_to_py(a))

                return f"{base_py}.{method_name}({', '.join(arg_strs)})"

            elif op[0] in ("array_subscript", "hash_subscript"):
                subscript_py = self.emit_subscript(op)
                return f"{base_py}{subscript_py}"

        return base_py

    def _expr_call(self, expr: tuple) -> str:
        """Convert function call expression to Python."""
        _, name, args = expr

        if isinstance(name, str):
            name = name.replace("::", ".")

        # Special case: ClassName.classMatch(obj, 'ClassName') -> isinstance(obj, ClassName)
        if name.endswith(".classMatch") and len(args) >= 2:
            obj_py = self.expr_to_py(args[0])
            class_name_py = self.expr_to_py(args[1])

            if isinstance(class_name_py, str):
                if (class_name_py.startswith('"') and class_name_py.endswith('"')) or \
                   (class_name_py.startswith("'") and class_name_py.endswith("'")):
                    class_name_str = class_name_py[1:-1]
                else:
                    class_name_str = class_name_py
            else:
                class_name_str = str(class_name_py).strip('"\'')

            return f"isinstance({obj_py}, {class_name_str})"

        arg_strings = []
        for a in args:
            if isinstance(a, tuple) and len(a) >= 2 and a[0] == "named_param":
                _, key_expr, val_expr = a
                if isinstance(key_expr, tuple) and key_expr[0] == "var":
                    key_str = key_expr[1]
                else:
                    key_str = self.expr_to_py(key_expr)
                val_str = self.expr_to_py(val_expr)
                arg_strings.append(f"{key_str} = {val_str}")
            else:
                arg_strings.append(self.expr_to_py(a))

        return f"{name}({', '.join(arg_strings)})"
