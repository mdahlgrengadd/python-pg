# DESCRIPTION
# Algebraic fraction answer requiring simplification
# ENDDESCRIPTION
# DBsubject(WeBWorK)
# DBchapter(WeBWorK tutorial)
# DBsection(PGML tutorial 2015)
# Date(06/01/2015)
# Institution(Hope College)
# Author(Paul Pearson)
# MO(1)
# KEYWORDS('algebra', 'algebraic fraction answer')
# :% name = Algebraic Fraction Answer
# :% type = Sample
# :% subject = PerlList([algebra, precalculus])
# :% categories = PerlList([fraction])
# :% section = preamble
#: Load the PODLINK('parserMultiAnswer.pl') macro to be able to consider two
#: answer rules together (the numerator and denominator) in the answer checker.
#: Note that the PODLINK('niceTables.pl') macro is implicitly used, and is
#: automatically loaded by the `PGML.pl` macro.
from pg.mathobjects import *
from pg.course import *
from pg.parser_multiAnswer import *
from pg.pgml import *
from pg.macros.core.pgml import PGML
from pg.standard import *
# Loaded: PGstandard.pl, PGML.pl, parserMultiAnswer.pl, PGcourse.pl

DOCUMENT()
# :% section = setup
#: Define MathObject formulas `$num` and `$den` which are the correct numerator
#: and denominator for the answer, as well as the bogus answers `$numbogus` and
#: `$denbogus` that result from not finding a common denominator (used in the
#: custom answer checker). A `MultiAnswer` is used to check the numerator and
#: denominator together.
#:
#: The `allowBlankAnswers => 1` option for the `MultiAnswer` object is set which
#: allows the answers to be left blank, so that partial credit can be given if
#: the student has the numerator or denominator correct but does not enter both.
#: This requires that the type of the students input be checked before using
#: those values in computations or warnings will be issued (in this case the
#: warning "Operands of '*' can't be words" is issued if
#: `$f1 * $f2stu == $f1stu * $f2` is computed). This is done for the
#: numerator, for example, with `Value::classMatch($f1stu, 'Formula')`.
#:
#: The student is also allowed to enter the fraction as either `(6y-3)/(y-2)` or
#: `(3-6y)/(2-y)`, since both are correct and it is not clear that one is
#: preferable to the other. For this the check `$f1 == $f1stu || -$f1 == $f1stu`
#: is used. Note that `||` is perl's "or" operator.
#:
#: Custom answer messages can be displayed by calling the `setMessage` method of
#: the `MultiAnswer` object that `$self` refers to.  For example, with
#: `$self->setMessage(1, "Simplify your answer further")`,
#: where 1 means to set the message for the first answer blank.
Context().variables.are(y='Real')
while True:
    a = random(2, 8, 2)
    b = random(3, 9, 2)
    c = random(1, 9, 1)
    if ((a * c) != b):
        break
num = Formula(f"{a} y - {b}")
den = Formula(f"y - {c}")
numbogus = Formula(f"{a}*y+{b}")
denbogus = Formula(f"(y-{c})*({c}-y)")


def _closure_checker_1(correct, student, self):
    f1stu, f2stu = student
    f1, f2 = correct
    if (((f1 == f1stu) and (f2 == f2stu)) or (((-f1) == f1stu) and ((-f2) == f2stu))):
        return [1, 1]
    elif ((f1 == f1stu) or ((-f1) == f1stu)):
        return [1, 0]
    elif (((numbogus == f1stu) or ((-numbogus) == f1stu)) or ((denbogus == f2stu) or ((-denbogus) == f2stu))):
        self.setMessage(1, "Find a common denominator first")
        self.setMessage(2, "Find a common denominator first")
        return [0, 0]
    elif ((f2 == f2stu) or ((-f2) == f2stu)):
        return [0, 1]
    elif ((isinstance(f1stu, Formula) and isinstance(f2stu, Formula)) and ((f1 * f2stu) == (f1stu * f2))):
        self.setMessage(1, "Simplify your answer further")
        self.setMessage(2, "Simplify your answer further")
        return [0, 0]
    else:
        return [0, 0]


multians = MultiAnswer(num, den).with_params(allowBlankAnswers=1, checker=_closure_checker_1
                                             )
# :% section = statement
#: The fraction answer is created using a `LayoutTable` from
#: PODLINK('niceTables.pl') via its `PGML` syntax. A `LayoutTable` is started
#: with `[#` and is ended with `#]*`. Options for the table are set in braces
#: after the ending `#]*`. Cells of the table are started wtih `[.` and ended
#: with `.]`. Options for a cell (some of which apply to the row as a whole)
#: are set in braces after the cell's ending `.]`. Rows of the table are ended
#: by a starred cell. For example `[. ... .]*`. Note that the second cell of
#: this table contains a nested `LayoutTable`.
#:
#: The outer `LayoutTable` has a single row with the mathematical expression and
#: then another `LayoutTable` with two rows that formats the fraction with a
#: bottom horizontal line under the first row. The padding is changed to improve
#: the look of the fraction.
PGML_BLOCK_0 = '''
Perform the indicated operations. Express your answer in reduced form.

[#
    [. [``\\frac{[a]y}{y - [c]} + \\frac{[b]}{[c] - y} =``] .]
    [.
        [#
            [. [_]{multians} .]*{ bottom => 1 }
            [. [_]{multians} .]
        #]*{ padding => [ 0.5, 0 ] }
    .]
#]*{ padding => [ 0, 0.5 ], valign => 'middle' }
'''
TEXT(PGML(PGML_BLOCK_0))
# :% section = solution
PGML_BLOCK_1 = '''
Solution explanation goes here.
'''
SOLUTION(PGML(PGML_BLOCK_1))
ENDDOCUMENT()

if __name__ == "__main__":
    """Execute problem and display results."""
    from pg.macros.core.pg_core import get_environment

    env = get_environment()
    if env:
        print("=" * 80)
        print("PROBLEM STATEMENT")
        print("=" * 80)
        print(''.join(env.output_array))

        if env.solution_array:
            print("\n" + "=" * 80)
            print("SOLUTION")
            print("=" * 80)
            print(''.join(env.solution_array))

        if env.hint_array:
            print("\n" + "=" * 80)
            print("HINT")
            print("=" * 80)
            print(''.join(env.hint_array))

        print("\n" + "=" * 80)
        print(f"ANSWERS: {len(env.answers_hash)} answer blank(s)")
        print("=" * 80)
        for name in sorted(env.answers_hash.keys()):
            print(f"  {name}")
