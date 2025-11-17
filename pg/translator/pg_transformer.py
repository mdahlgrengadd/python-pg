"""
AST Transformer for PG/Perl parse trees.

This module provides the Lark Transformer that converts parsed
Perl-like syntax into an intermediate representation (IR) that can
be emitted as Python code.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lark import Transformer

# Try importing Lark. If unavailable, define dummy stand-ins.
try:
    from lark import Transformer, v_args
    _LARK_AVAILABLE = True
except Exception:
    _LARK_AVAILABLE = False

    def v_args(*args, **kwargs):
        def wrapper(func):
            return func
        return wrapper

    class Transformer:
        pass


def create_pg_transformer() -> Transformer:
    """
    Create and return a Lark transformer that lowers parse trees to IR.

    The transformer converts Lark parse trees from the PG grammar into
    an intermediate representation (IR) consisting of tuples. This IR
    is designed to be easily converted to Python code.

    IR formats:
        - ("call", name, args) - Function call
        - ("assign", var, expr) - Variable assignment
        - ("if", cond, block, clauses) - If statement
        - ("while", cond, block) - While loop
        - ("for", var, expr, block) - For loop
        - ("bin", left, op, right) - Binary expression
        - ("ternary", cond, true_val, false_val) - Ternary operator
        - ("postfix", expr, op) - Method call or subscript
        - ("var", name) - Variable reference
        - ("closure", body) - Sub closure
        And many more...

    Returns:
        Transformer: Lark transformer instance for converting AST to IR

    Raises:
        RuntimeError: If Lark is not available
    """
    if not _LARK_AVAILABLE:
        raise RuntimeError("Lark parser is not available. Cannot create transformer.")

    @v_args(inline=True)
    class ToIR(Transformer):
        """Lower the parse tree into intermediate representation (IR)."""

        def start(self, *stmts):
            """Flatten the list of statements from the top-level start rule."""
            return list(stmts)

        def stmt(self, item):
            """Unwrap the single child of a stmt production."""
            return item

        # Document control
        def document_call(self):
            return ("call", "DOCUMENT", [])

        def enddocument_call(self):
            return ("call", "ENDDOCUMENT", [])

        def loadmacros_call(self, *args):
            return ("noop", )

        # Control flow statements
        def if_stmt(self, condition, block, *clauses):
            """Lower if statement with optional elsif/else clauses."""
            return ("if", condition, block, list(clauses))

        def elsif_clause(self, condition, block):
            return ("elsif", condition, block)

        def else_clause(self, block):
            return ("else", block)

        def unless_stmt(self, condition, block):
            """Lower unless statement (if not)."""
            return ("unless", condition, block)

        def while_stmt(self, condition, block):
            """Lower while loop."""
            return ("while", condition, block)

        def for_stmt(self, var, expr, block):
            """Lower for loop."""
            return ("for", var, expr, block)

        def foreach_stmt(self, var, expr, block):
            """Lower foreach loop."""
            return ("foreach", var, expr, block)

        def do_until_stmt(self, block, condition):
            """Lower do-until loop."""
            return ("do_until", block, condition)

        def block(self, *stmts):
            """Lower block of statements."""
            return ("block", list(stmts))

        # Statement modifiers
        def stmt_modifier(self, child):
            """Pass through stmt_modifier_if or stmt_modifier_unless."""
            return child

        def stmt_modifier_if(self, stmt, condition):
            """Lower statement modifier with 'if'."""
            return ("stmt_modifier", stmt, "if", condition)

        def stmt_modifier_unless(self, stmt, condition):
            """Lower statement modifier with 'unless'."""
            return ("stmt_modifier", stmt, "unless", condition)

        def simple_stmt(self, stmt):
            return stmt

        # Assignments
        def assign_stmt(self, var, expr):
            """Lower a variable declaration or assignment."""
            return ("assign", var, expr)

        def subscript_assign(self, var, subscript, expr):
            """Lower subscript assignment: $arr[0] = value."""
            return ("subscript_assign", var, subscript, expr)

        # Subscripting
        def array_subscript(self, expr):
            return ("array_subscript", expr)

        def hash_subscript(self, expr):
            return ("hash_subscript", expr)

        # Expressions
        def call_expr(self, name, *args):
            """Lower a function call expression."""
            arglist = args[0] if args else []
            # Convert Perl namespace operator :: to Python dot notation
            name = name.replace("::", ".") if isinstance(name, str) else name
            return ("call", name, arglist)

        def expr_stmt(self, expr):
            """Lower an expression statement."""
            return ("expr", expr)

        def ternary_expr(self, *parts):
            """Lower ternary operator: cond ? true : false."""
            if len(parts) == 1:
                return parts[0]
            elif len(parts) == 3:
                cond, true_val, false_val = parts
                return ("ternary", cond, true_val, false_val)
            return parts[0]

        def binary_expr(self, left, *rest):
            """Lower binary operations."""
            expr = left
            for op, right in zip(rest[::2], rest[1::2]):
                # Extract operator string from Tree or Token
                if hasattr(op, 'children') and op.children:
                    # If op is a Tree with children, get the first child
                    op_tok = op.children[0]
                elif hasattr(op, 'data'):
                    # If op is a Tree without children (inline rules), use a Token
                    # This shouldn't happen with properly defined grammars
                    op_tok = op
                else:
                    # op is already a Token or string
                    op_tok = op

                # Extract the actual operator string value
                if hasattr(op_tok, 'value'):
                    op_str = op_tok.value
                elif hasattr(op_tok, 'type'):
                    # Token without value attribute
                    op_str = str(op_tok)
                else:
                    op_str = str(op_tok)

                expr = ("bin", expr, op_str, right)
            return expr

        # Operator extractors - these are needed because operators are defined as rules
        def add_op(self, token):
            """Extract add operator token."""
            # token is now a Token object from the terminal
            if hasattr(token, 'value'):
                return token.value
            else:
                return str(token)

        def mul_op(self, token):
            """Extract mul operator token."""
            # token is now a Token object from the terminal
            if hasattr(token, 'value'):
                return token.value
            else:
                return str(token)

        def comp_op(self, token):
            """Extract comparison operator token."""
            # Tokens like EQ, GT, etc. come through as Token objects
            if hasattr(token, 'type'):
                # Map token types to operator strings
                op_map = {
                    'EQ': 'eq', 'NE': 'ne', 'LT': 'lt', 'GT': 'gt',
                    'LE': 'le', 'GE': 'ge', 'EQEQ': '==', 'BANGEQ': '!=',
                    'LANGLE': '<', 'RANGLE': '>', 'LTEQ': '<=', 'GTEQ': '>='
                }
                return op_map.get(token.type, token.value)
            if hasattr(token, 'value'):
                return token.value
            return str(token)

        def range_expr(self, *parts):
            """Lower range operator: 0..10."""
            if len(parts) == 2:
                start, end = parts
                return ("range", start, end)
            return parts[0]

        def unary_minus(self, expr):
            return ("unary", "-", expr)

        def unary_not(self, expr):
            return ("unary", "!", expr)

        def postfix_expr(self, primary, *postfix_ops):
            """Lower postfix operations (method calls, subscripts)."""
            expr = primary
            for op in postfix_ops:
                expr = ("postfix", expr, op)
            return expr

        def postfix_op(self, child):
            """Pass through the postfix operation (method_call or subscript)."""
            return child

        def method_call(self, name, *args):
            """Lower method call: ->method()."""
            arglist = args[0] if args else []
            return ("method_call", name, arglist)

        # Map and grep
        def map_expr(self, block_expr, list_expr):
            """Lower map block."""
            return ("map", block_expr, list_expr)

        def grep_expr(self, block_expr, list_expr):
            """Lower grep block."""
            return ("grep", block_expr, list_expr)

        # Regex
        def regex_literal(self, *args):
            """Lower regex literal: qr/pattern/flags."""
            # Can receive (qr_token, pattern, flags) or (pattern, flags) depending on parsing
            if len(args) == 3:
                qr, pattern, flags = args
            elif len(args) == 2:
                pattern, flags = args
            else:
                # Fallback
                pattern = args[0] if args else ""
                flags = ""
            return ("regex", str(pattern), str(flags))

        # Tokens
        def VAR(self, tok):
            return ("var", str(tok))

        def NAME(self, tok):
            return str(tok)

        def NUMBER(self, tok):
            return str(tok)

        def STRING(self, tok):
            return str(tok)

        def args(self, *items):
            return list(items)

        def arg_item(self, *children):
            """Handle argument item: expr or expr => expr"""
            if len(children) == 1:
                # Just an expression
                return children[0]
            elif len(children) == 2:
                # expr => expr (named parameter)
                key_expr, val_expr = children
                return ("named_param", key_expr, val_expr)
            else:
                return children[0]

        def var(self, child):
            """Unwrap the var rule to return its child."""
            return child

        def atom(self, child):
            """Unwrap the atom rule to return its child."""
            return child

        # Sub closures
        def sub_closure(self, body):
            """Lower sub { ... } closure."""
            return ("closure", body)

        def closure_body(self, *stmts):
            """Lower closure body statements."""
            return list(stmts)

        def closure_stmt(self, stmt):
            """Unwrap closure statement."""
            return stmt

        # Parameter unpacking
        def param_decl(self, var_list, deref):
            """Lower my ($a, $b) = @_; or my ($a, $b) = @$arr;"""
            return ("param_unpack", var_list, deref)

        def param_var_list(self, *vars):
            """Lower parameter variable list."""
            return list(vars)

        # Array dereferencing in parameter context
        def deref_args(self):
            """Lower @_ dereference."""
            return ("special_var", "@_")

        def deref_var(self, name):
            """Lower @$var dereference."""
            return ("array_deref", ("var", f"${name}"))

        # Return statements
        def return_stmt(self, value):
            """Lower return statement."""
            return ("return", value)

        def return_expr_special(self, expr):
            """Unwrap return expression special (handles both arrays and regular exprs)."""
            return expr

        def return_array(self, expr_list):
            """Lower return [...];"""
            # expr_list is the result of return_list which is a list of expressions
            return ("array", expr_list if isinstance(expr_list, list) else [expr_list])

        def return_expr(self, expr):
            """Lower return expr;"""
            return expr

        def return_list(self, *exprs):
            """Lower comma-separated return list."""
            return list(exprs)

        def return_empty_array(self):
            """Lower return [] (empty array)."""
            return ("array", [])

        # Array dereferencing in expressions
        def array_deref_var(self, name):
            """Lower @$name array dereference."""
            return ("array_deref", ("var", f"${name}"))

        def array_deref_args(self):
            """Lower @_ array dereference."""
            return ("special_var", "@_")

    return ToIR()
