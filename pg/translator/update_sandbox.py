#!/usr/bin/env python3
"""Update sandbox.py with all needed fixes."""

import re

# Read current sandbox.py
with open('pg_translator/sandbox.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: Add macro imports
old_import = '''# Import PG packages (these must be installed in the Python environment)
try:
    from pg.math import Real, Complex, Formula, Point, Vector, Matrix, Interval
    from pg.parser import Context, Parser
    from pg.answer import AnswerEvaluator
except ImportError as e:
    print(json.dumps({"success": False, "errors": f"Import error: {e}"}))
    sys.exit(1)

# Compute is an alias for Formula (for Perl compatibility)
Compute = Formula

# Set random seed
random.seed({seed})'''

new_import = '''# Import PG packages (these must be installed in the Python environment)
try:
    from pg.math import Real, Complex, Formula, Point, Vector, Matrix, Interval
    from pg.parser import Context, Parser
    from pg.answer import AnswerEvaluator
    from pg.answer.evaluators.numeric import NumericEvaluator
    from pg.answer.evaluators.formula import FormulaEvaluator
    from pg.answer.evaluators.string import StringEvaluator
except ImportError as e:
    print(json.dumps({"success": False, "errors": f"Import error: {e}"}))
    sys.exit(1)

# Compute is an alias for Formula (for Perl compatibility)
Compute = Formula

# Set random seed
random.seed({seed})

# Macro functions (Perl compatibility)
def num_cmp(correct, **options):
    """Numeric comparison answer evaluator."""
    return NumericEvaluator(correct_answer=correct, **options)

def fun_cmp(correct, **options):
    """Formula/function answer evaluator."""
    return FormulaEvaluator(correct_answer=correct, **options)

def str_cmp(correct, **options):
    """String comparison answer evaluator."""
    return StringEvaluator(correct_answer=correct, **options)'''

content = content.replace(old_import, new_import)

# Fix 2: Add random() function wrapper after PGEnv class
# Find where to insert (after pg_env = PGEnv())
insert_pos = content.find('pg_env = PGEnv()\n') + len('pg_env = PGEnv()\n')
random_func = '''
# PG random function (Perl compatibility)
def random_func(min_val=0.0, max_val=1.0):
    """Generate random number in range [min_val, max_val)."""
    return min_val + random.random() * (max_val - min_val)

# Make 'random' callable as a function (not just the module)
# Store original module
_random_module = random
# Replace with our wrapper but keep module methods available
class RandomWrapper:
    def __call__(self, min_val=0.0, max_val=1.0):
        return random_func(min_val, max_val)
    def __getattr__(self, name):
        return getattr(_random_module, name)
random = RandomWrapper()

'''

content = content[:insert_pos] + random_func + content[insert_pos:]

# Fix 3: Update ANS() to accept optional name parameter
old_ans = '''def ANS(*evaluators):
    """Register answer evaluators."""
    for i, evaluator in enumerate(evaluators):
        ans_name = f"AnSwEr{{len(pg_env.answers):04d}}"
        pg_env.answers[ans_name] = str(evaluator)  # Serialize evaluator'''

new_ans = '''def ANS(*args):
    """Register answer evaluators (with optional name)."""
    # Support both ANS(evaluator) and ANS(evaluator, name)
    if len(args) == 1:
        evaluator = args[0]
        ans_name = f"AnSwEr{{len(pg_env.answers):04d}}"
        pg_env.answers[ans_name] = evaluator  # Store evaluator object
    elif len(args) == 2:
        evaluator, ans_name = args
        pg_env.answers[ans_name] = evaluator  # Store evaluator object
    else:
        for evaluator in args:
            ans_name = f"AnSwEr{{len(pg_env.answers):04d}}"
            pg_env.answers[ans_name] = evaluator  # Store evaluator object'''

content = content.replace(old_ans, new_ans)

# Fix 4: Update NAMED_ANS to store evaluator object
old_named = '''def NAMED_ANS(name, evaluator):
    """Register named answer evaluator."""
    pg_env.answers[name] = str(evaluator)'''

new_named = '''def NAMED_ANS(name, evaluator):
    """Register named answer evaluator."""
    pg_env.answers[name] = evaluator  # Store evaluator object'''

content = content.replace(old_named, new_named)

# Write updated content
with open('pg_translator/sandbox.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Updated sandbox.py with:")
print("  - Macro imports (num_cmp, fun_cmp, str_cmp)")
print("  - random() function wrapper")
print("  - ANS() with optional name parameter")
print("  - Store evaluator objects (not strings)")
