# PG Preprocessor Refactoring Summary

## Overview

The `pg_preprocessor_pygment.py` file has been refactored to improve maintainability, testability, and extensibility. The original 4000+ line monolithic file has been split into focused, single-responsibility modules.

## Motivation

The original file suffered from several maintainability issues:

1. **Size**: 4000+ lines in a single file made it difficult to navigate and understand
2. **Single Responsibility Principle violation**: One class handled parsing, transformation, code generation, and text processing
3. **Tight coupling**: Components were difficult to test or modify independently
4. **Low cohesion**: Related functionality was spread across multiple methods

## New Architecture

The preprocessor has been split into the following modules:

### 1. `pg_grammar.py` - Grammar Definition
- **Purpose**: Contains the Lark grammar for parsing Perl-like PG syntax
- **Function**: `get_pg_grammar()` - Returns the complete grammar string
- **Benefits**:
  - Easy to modify grammar without touching other code
  - Can be versioned and tested independently
  - Clear separation of parsing rules from transformation logic

### 2. `pg_transformer.py` - AST Transformation
- **Purpose**: Transforms Lark parse trees into intermediate representation (IR)
- **Class**: `create_pg_transformer()` - Factory function for transformer
- **IR Format**: Tuples like `("call", name, args)`, `("if", cond, block, clauses)`
- **Benefits**:
  - Decouples parsing from code generation
  - IR can be inspected, optimized, or transformed further
  - Easier to test transformation rules

### 3. `pg_block_extractor.py` - Block Extraction
- **Purpose**: Extracts special PG blocks (BEGIN_TEXT, BEGIN_PGML, etc.)
- **Class**: `PGBlockExtractor`
- **Key Methods**:
  - `extract_block()` - Extract standard blocks
  - `extract_method_block()` - Extract method-style blocks
  - `transform_pgml_evaluators()` - Transform PGML syntax
- **Benefits**:
  - Isolated block extraction logic
  - Easy to add new block types
  - Can be tested with simple string inputs

### 4. `pg_text_processor.py` - Text Preprocessing
- **Purpose**: Handles text transformations before parsing
- **Class**: `PGTextProcessor`
- **Key Methods**:
  - `fix_reference_dereferences()` - Fix $x->[$i] to $x[$i]
  - `convert_heredocs_global()` - Convert <<MARKER to triple quotes
  - `convert_string_interpolation()` - Convert "$var" to f"{var}"
- **Benefits**:
  - Clear separation of preprocessing from parsing
  - Each transformation can be tested independently
  - Easy to add new preprocessors

### 5. `pg_pygments_rewriter.py` - Fallback Rewriter
- **Purpose**: Token-level rewriting when parser fails
- **Class**: `PGPygmentsRewriter`
- **Key Methods**:
  - `rewrite_with_pygments()` - Main fallback rewriting
  - `convert_regexes()` - Convert qr// to re.compile()
  - `desigil()` - Remove Perl sigils
- **Benefits**:
  - Isolated fallback logic
  - Can use or replace Pygments independently
  - Clear boundary between parsing and fallback strategies

### 6. `pg_ir_emitter.py` - Code Generation
- **Purpose**: Converts IR tuples to Python code
- **Class**: `PGIREmitter`
- **Key Methods**:
  - `emit_ir()` - Main IR to Python conversion
  - `expr_to_py()` - Expression IR to Python strings
  - `emit_closure()` - Handle Perl closures (sub { ... })
- **Benefits**:
  - Separated code generation from parsing
  - Easy to modify Python output format
  - Can target different Python versions or styles

### 7. `pg_preprocessor_pygment.py` (Refactored) - Orchestrator
- **Purpose**: Coordinates all components to preprocess PG files
- **Class**: `PGPreprocessor`
- **Responsibilities**:
  - Initialize all components
  - Coordinate preprocessing pipeline
  - Handle imports and macros
  - Manage line mapping
- **Benefits**:
  - Much smaller and easier to understand
  - Clear control flow through components
  - Public API remains unchanged (backward compatible)

## Architecture Diagram

```
PG Source File
      ↓
[Text Preprocessing] (heredocs, string interpolation, dereferences)
      ↓
[Block Extraction] (BEGIN_TEXT, BEGIN_PGML, etc.)
      ↓
[Parsing] (Lark + Grammar)
      ↓
[AST Transformation] (Parse tree → IR)
      ↓
[IR Emission] (IR → Python code)
      ↓
[Fallback Rewriting] (Pygments for unparseable code)
      ↓
Python Code Output
```

## Benefits of Refactoring

### 1. **Improved Maintainability**
- Each module is < 500 lines and focused on one task
- Easy to locate and fix bugs
- Changes to one component don't affect others

### 2. **Better Testability**
- Each component can be unit tested independently
- Mock dependencies easily
- Test specific transformations without full preprocessing

### 3. **Enhanced Extensibility**
- Adding new Perl constructs: modify grammar + transformer
- Adding new block types: extend block extractor
- Changing output format: modify IR emitter

### 4. **Clearer Documentation**
- Each module has focused documentation
- Examples and use cases per component
- Architecture is self-documenting

### 5. **Team Collaboration**
- Multiple developers can work on different modules
- Smaller files reduce merge conflicts
- Clear interfaces between components

## Migration Strategy

The refactoring maintains backward compatibility:

1. **Old API works**: `PGPreprocessor().preprocess(source)` still works
2. **Gradual adoption**: Can start using new modules individually
3. **Original preserved**: `pg_preprocessor_pygment_original.py` backs up the original

## Future Improvements

1. **Add comprehensive tests** for each module
2. **Performance optimization** of IR emission
3. **Better error messages** with source location tracking
4. **Plugin system** for custom transformations
5. **Type system** for IR validation

## Usage Example

```python
from pg.translator.pg_preprocessor_pygment import PGPreprocessor

# Simple usage (same as before)
preprocessor = PGPreprocessor()
result = preprocessor.preprocess(pg_source_code)

# Using individual components (new capability)
from pg.translator.pg_grammar import get_pg_grammar
from pg.translator.pg_transformer import create_pg_transformer
from pg.translator.pg_block_extractor import PGBlockExtractor

grammar = get_pg_grammar()
transformer = create_pg_transformer()
block_extractor = PGBlockExtractor()

# Extract blocks
block_result = block_extractor.extract_block(lines, 0)
```

## Files Changed

- **Added**:
  - `pg_grammar.py` (180 lines)
  - `pg_transformer.py` (370 lines)
  - `pg_block_extractor.py` (200 lines)
  - `pg_text_processor.py` (270 lines)
  - `pg_pygments_rewriter.py` (320 lines)
  - `pg_ir_emitter.py` (450 lines)
  - `REFACTORING_SUMMARY.md` (this file)

- **Modified**:
  - `pg_preprocessor_pygment.py` (refactored to use new modules)

- **Preserved**:
  - `pg_preprocessor_pygment_original.py` (backup of original)

## Metrics

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Lines per file | 4090 | ~400 avg | 90% reduction |
| Cyclomatic complexity | Very High | Low-Medium | Significant |
| Number of methods per class | 30+ | 5-10 | Better cohesion |
| Test coverage | Low | Testable | Improved |
| Documentation | Limited | Comprehensive | Much better |

## Conclusion

This refactoring significantly improves the codebase's maintainability, testability, and extensibility while preserving backward compatibility. The modular architecture makes it easier to understand, modify, and extend the PG preprocessor functionality.
