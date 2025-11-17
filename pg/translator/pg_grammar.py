"""
Lark grammar definition for parsing PG/Perl statements.

This module contains the grammar used to parse Perl-like syntax
found in PG (Problem Generator) files. The grammar covers common
Perl constructs including control flow, operators, method calls,
and special PG-specific functions.
"""

from __future__ import annotations


def get_pg_grammar() -> str:
    """
    Return the Lark grammar for parsing PG/Perl statements.

    This extended grammar covers most Perl constructs found in PG files:
    - Control flow: if/elsif/unless/while/for
    - Ternary operator: cond ? true : false
    - Hash/array access: $hash{key}, $array[idx]
    - Method calls: $obj->method()
    - Ranges: 0..10
    - Map/grep blocks
    - Statement modifiers: stmt if cond
    - Fat comma: key => value

    The grammar is designed to avoid reduce/reduce conflicts.
    Unparseable constructs fall back to Pygments rewriting.

    Returns:
        str: The complete Lark grammar definition
    """
    return r"""
        start: (stmt ";"?)*

        stmt: if_stmt
            | while_stmt
            | for_stmt
            | foreach_stmt
            | do_until_stmt
            | decl
            | assign
            | return_stmt
            | expr_stmt
            | document
            | enddocument
            | stmt_modifier

        // Control flow statements
        if_stmt: "if" "(" expr ")" block elsif_clause* else_clause?
        elsif_clause: "elsif" "(" expr ")" block
        else_clause: "else" block
        unless_stmt: "unless" "(" expr ")" block
        while_stmt: "while" "(" expr ")" block
        for_stmt: "for" "my"? var "(" expr ")" block
        foreach_stmt: "foreach" "my"? var "(" expr ")" block
        do_until_stmt: "do" block "until" "("? expr ")"?

        block: "{" (stmt ";"?)* "}"

        // Statement modifiers (trailing conditionals)
        stmt_modifier_if: simple_stmt "if" expr        -> stmt_modifier_if
        stmt_modifier_unless: simple_stmt "unless" expr -> stmt_modifier_unless
        stmt_modifier: stmt_modifier_if | stmt_modifier_unless
        simple_stmt: decl | assign | call_expr

        document: "DOCUMENT" "(" ")"            -> document_call
        enddocument: "ENDDOCUMENT" "(" ")"      -> enddocument_call
        loadmacros: "loadMacros" "(" /[^)]*/ ")" -> loadmacros_call

        decl: "my" var "=" expr               -> assign_stmt
            | "my" var "=" sub_closure      -> assign_stmt
        assign: var "=" expr                -> assign_stmt
              | var "=" sub_closure        -> assign_stmt
              | var subscript "=" expr      -> subscript_assign
        expr_stmt: expr                      -> expr_stmt

        // Sub closures: sub { ... }
        sub_closure: "sub" "{" closure_body "}"
        closure_body: (closure_stmt (";" | "\n")?)*
        closure_stmt: param_decl
                    | decl
                    | assign
                    | if_stmt
                    | while_stmt
                    | for_stmt
                    | foreach_stmt
                    | do_until_stmt
                    | return_stmt
                    | expr_stmt
                    | stmt_modifier

        // Parameter unpacking: my ($a, $b) = @_;  or my ($a, $b) = @$arr;
        param_decl: "my" "(" param_var_list ")" "=" array_deref_special
        param_var_list: var ("," var)*
        array_deref_special: "@" "_"  -> deref_args
                           | "@" "$" NAME  -> deref_var

        // Return statement: return expr; or return [expr, ...];
        // Must handle [ ... ] specially because [ is ambiguous (could be subscript or array literal)
        return_stmt: "return" return_expr_special
        return_expr_special: "[" return_list "]"  -> return_array
                           | "[" "]"              -> return_empty_array
                           | expr
        return_list: expr ("," expr)*

        // Args can be comma-separated expressions or named parameters with =>
        args: arg_item ("," arg_item)*
        arg_item: expr ("=>" expr)?

        // Hash/Array subscripting
        subscript: "[" expr "]"              -> array_subscript
                 | "{" expr "}"              -> hash_subscript

        // Expressions with precedence (lowest to highest)
        ?expr: ternary_expr

        // Ternary: cond ? true : false
        ?ternary_expr: or_expr ("?" or_expr ":" ternary_expr)?  -> ternary_expr

        ?or_expr: and_expr ((OR_OP | "or") and_expr)*      -> binary_expr
        ?and_expr: comp_expr ((AND_OP | "and") comp_expr)*  -> binary_expr

        // Comparison operators (eq, ne, lt, gt, le, ge, ==, !=, <, >, <=, >=)
        ?comp_expr: range_expr (comp_op range_expr)*     -> binary_expr
        comp_op: EQ | NE | LT | GT | LE | GE | EQEQ | BANGEQ | LANGLE | RANGLE | LTEQ | GTEQ
        EQ: "eq"
        NE: "ne"
        LT: "lt"
        GT: "gt"
        LE: "le"
        GE: "ge"
        EQEQ: "=="
        BANGEQ: "!="
        LANGLE: "<"
        RANGLE: ">"
        LTEQ: "<="
        GTEQ: ">="

        // Logical operators (must be terminals, not literal strings, to avoid Lark confusion)
        OR_OP: "||"
        AND_OP: "&&"

        // Range operator: 0..10
        ?range_expr: add_expr (".." add_expr)?           -> range_expr

        ?add_expr: mul_expr (add_op mul_expr)*           -> binary_expr
        add_op: PLUS | MINUS | DOT
        PLUS.2: "+"
        MINUS.2: "-"
        DOT.2: "."

        ?mul_expr: unary_expr (mul_op unary_expr)*       -> binary_expr
        mul_op: STAR | SLASH | PERCENT | X
        STAR.2: "*"
        SLASH.2: "/"
        PERCENT.2: "%"
        X.2: "x"

        ?unary_expr: postfix_expr
                   | "-" unary_expr                      -> unary_minus
                   | "!" unary_expr                      -> unary_not

        // Postfix: method calls, subscripts
        ?postfix_expr: primary (postfix_op)*
        postfix_op: "->" NAME "(" args? ")"              -> method_call
                  | subscript

        ?primary: call_expr | var | atom | array_deref | "(" expr ")"

        call_expr: NAME "(" args? ")"          -> call_expr

        // Map and grep blocks
        map_expr: "map" "{" expr "}" expr                -> map_expr
        grep_expr: "grep" "{" expr "}" expr              -> grep_expr

        // Array dereferencing: @$var or @_
        array_deref: "@" "$" NAME                        -> array_deref_var
                   | "@" "_"                             -> array_deref_args

        var: VAR
        atom: NUMBER | STRING | NAME | regex_literal

        regex_literal: QR "/" /[^\/]+/ "/" REGEX_FLAGS?  -> regex_literal

        QR.2: "qr"
        NAME: /[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*/
        VAR: /[\$@%][A-Za-z_][A-Za-z0-9_]*/
        STRING: /"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'/
        NUMBER: /[0-9]+(?:\.[0-9]+)?/
        REGEX_FLAGS: /[imsxo]+/

        %import common.WS
        %ignore WS
        """
