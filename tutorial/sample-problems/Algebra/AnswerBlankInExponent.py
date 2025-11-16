## DESCRIPTION
## Answer blank in the exponent
## ENDDESCRIPTION
## DBsubject(WeBWorK)
## DBchapter(WeBWorK tutorial)
## DBsection(PGML tutorial 2015)
## Date(06/01/2015)
## Institution(Hope College)
## Author(Paul Pearson)
## MO(1)
## KEYWORDS('algebra', 'answer blank in the exponent')
#:% name = Answer Blank in the Exponent
#:% type = Sample
#:% subject = PerlList([algebra, precalculus])
#:% categories = PerlList([exponent])
#:% section = preamble
from pg.mathobjects import *
from pg.course import *
from pg.pgml import *
from pg.standard import *
from pg.macros.core.pgml import PGML
# Loaded: PGstandard.pl, PGML.pl, PGcourse.pl

DOCUMENT()
#:% section = setup
#: Set the context to allow only the variables `a` and `b` and choose a random
#: exponent.
#:
#: The exponential layout is in HTML using a pair of adjacent `span` elements
#: with the right one shifted up using the CSS style `vertical-align`.
#: In hardcopy mode, a LaTeX exponent is used.
Context().variables.are(a = 'Real', b = 'Real')
n = random(3, 9)
# TeX
expression = f"a^{{{n}}} b^{{{n}}}"
# MathObjects
base = Formula("a*b")
exponent = Formula(f"{n}")
# Display exponents nicely
if (displayMode == 'TeX'):
    exp = f"\\( \\displaystyle {expression} = ("  +  ans_rule(4)  +  ")^{"  +  ans_rule(4)  +  "}\\)"
else:
    exp = f"<span>\\(\\displaystyle {expression} = \\Big(\\)"  +  ans_rule(4)  +  '\\(\\Big)\\)</span><span style="vertical-align: 12pt;">'  +  ans_rule(4)  +  '</span>'
#:% section = statement
#: Insert the exponential stored as `$exp`.
PGML_BLOCK_0 = '''
Rewrite the following using a single exponent.

[$exp]*
'''
TEXT(PGML(PGML_BLOCK_0))
#:% section = answer
#: It is necessary to install the answer evaluator with `ANS`
#: since `ans_rule` was used to produce answer blanks.
ANS(base.cmp())
ANS(exponent.cmp())
#:% section = solution
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
