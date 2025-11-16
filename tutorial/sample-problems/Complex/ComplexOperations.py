## DESCRIPTION
## This demonstrates basic operations with complex numbers.
## ENDDESCRIPTION
## DBsubject(WeBWorK)
## DBchapter(WeBWorK tutorial)
## DBsection(Problem Techniques)
## Date(06/01/2023)
## Institution(Fitchburg State University)
## Author(Peter Staab)
## MO(1)
## KEYWORDS('complex','addition','subtraction','absolute value')
#:% name = Basic Operations of Complex numbers
#:% type = [technique]
#:% subject = [complex]
#:% categories = [complex variables]
#:% section = preamble
from pg.mathobjects import *
from pg.course import *
from pg.pgml import *
from pg.standard import *
# Loaded: PGstandard.pl, PGML.pl, PGcourse.pl

DOCUMENT()
#:% section = setup
#: To use complex numbers, switch to the `Complex` context with
#: `Context('Complex')`.
#:
#: Several ways to create a complex number that are demonstrated. Notice for the
#: 4th example that `i` is defined as a MathObject complex number that can be
#: used directly in Perl computations.
Context('Complex')
z0 = Complex(non_zero_random((-5), 4), non_zero_random((-5), 5))
z1 = Complex([ -1, 4 ])
z2 = Complex("2-4i")
z3 = 3 - (4 * i)
ans1 = z0 + z1
a0 = non_zero_random((-4), 4)
a1 = random(1, 5)
ans2 = Compute(f"{a0}*{z1}-{a1}*{z2}")
#:% section = statement
#: For the last three answer rules, the correct answer is directly computed from
#: previously defined variables in the answer rule option braces `{...}` instead
#: of being stored in another variable, as in the first two answer rules. Either
#: method is correct. Usually you would only need store the answer in a variable
#: if it will be used in other places in the code as well which is not done even
#: for the first two answers rules in this case.
#:
#: Note that the `**` in the last answer is the Perl exponent operator.
PGML_BLOCK_0 = '''
Let = {}
Let [`z_0 = [$z0]`], [`z_1 = [$z1]`], [`z_2 = [$z2]`] and [`z_3 = [$z3]`]. Find

[`z_0+z_1 =`] [___]{ans1}

[`[$a0]z_1 - [$a1]z_2 =`] [_____]{ans2}

[`z_1 z_2 =`] [___]{z1*z2}

[``\\frac{z_3}{z_0} =``] [___]{z3/z0}

[``z_2^2 =``] [__]{z2**2}
'''
TEXT(PGML(PGML_BLOCK_0))
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
