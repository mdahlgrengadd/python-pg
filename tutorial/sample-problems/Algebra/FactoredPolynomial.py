## DESCRIPTION
## Factored polynomial
## ENDDESCRIPTION
## DBsubject(WeBWorK)
## DBchapter(WeBWorK tutorial)
## DBsection(PGML tutorial 2015)
## Date(06/01/2015)
## Institution(Hope College)
## Author(Paul Pearson)
## Static(1)
## MO(1)
## KEYWORDS('algebra', 'factored polynomial')
#:% name = Factored Polynomial
#:% type = Sample
#:% subject = PerlList([algebra, precalculus])
#:% categories = PerlList([polynomials])
#:% section = preamble
#: Additional contexts provided by the PODLINK('contextPolynomialFactors.pl')
#: and PODLINK('contextLimitedPowers.pl') macros are needed.
from pg.mathobjects import *

DOCUMENT()
#:% section = setup
#: Before computing the answer which will be the factored form of the
#: polynomial, change to the `PolynomialFactors-Strict` context, and restrict
#: the allowed powers to only 0 and 1 using the `LimitedPowers::OnlyIntegers`
#: method. Note that restricting all powers to 0 or 1 means that repeated
#: factors will have to be entered in the form `k(ax+b)(ax+b)` instead of
#: `k(ax+b)^2`. Also, restricting all exponents to 0 or 1 means that the
#: polynomial must factor as a product of linear factors (no irreducible
#: quadratic factors can appear). If the exponents of 0, 1, or 2 were allowed,
#: then students would be allowed to enter reducible quadratic factors. There
#: are no restrictions on the coefficients, so the quadratic could have any
#: nonzero leading coefficient. Also set `singleFactors => 0` so that repeated,
#: non-simplified factors do not generate errors.
# Expanded form
Context('Numeric')
poly = Compute('8x^2 + 28x + 12')
# Factored form
Context('PolynomialFactors-Strict')
Context().flags.set(singleFactors = 0)
LimitedPowers.OnlyIntegers( minPower = 0, maxPower = 1, message = 'either 0 or 1', )
factored = Compute('4(2x+1)(x+3)')
#:% section = statement
#: Explicitly inform students to enter answers in the form `k(ax+b)(cx+d)`.
PGML_BLOCK_0 = '''
Write the quadratic expression [`[$poly]`] in factored form [`k(ax+b)(cx+d)`].

[_]{factored}{20}
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
