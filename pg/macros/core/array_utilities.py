"""
Array and Data Structure Utilities for WeBWorK.

This module provides functions for manipulating arrays, lists, and other
data structures used in PG problems. These are Perl-compatibility functions
that work with Python's list data structures.

Based on Perl array functions and WeBWorK macro libraries.
"""

from typing import Any, List, Optional


def splice(array: List[Any], offset: int, length: int = 1, 
           replacement: Optional[Any] = None) -> Any:
    """
    Splice an array - remove and replace elements.
    
    Similar to Perl's splice function. Removes length elements from the array
    starting at offset and optionally replaces them with new elements.
    Returns the removed element(s).
    
    Args:
        array: Array/list to splice
        offset: Starting position (can be negative for offset from end)
        length: Number of elements to remove (default 1)
        replacement: Element(s) to insert in place of removed items
        
    Returns:
        The removed element (or None if array is invalid)
        
    Perl Source: Perl splice(@array, offset, length, replacement)
    """
    if not isinstance(array, list):
        return None

    # Handle default length
    if length is None or length == 1:
        length = 1

    # Get the elements to remove
    if offset < 0:
        offset = len(array) + offset

    if offset < 0 or offset >= len(array):
        return None

    # Remove and return the element
    removed = array.pop(offset)

    # Handle replacement if provided
    if replacement is not None:
        if isinstance(replacement, (list, tuple)):
            for i, item in enumerate(replacement):
                array.insert(offset + i, item)
        else:
            array.insert(offset, replacement)

    return removed


def push(array: List[Any], *items: Any) -> Optional[int]:
    """
    Push function - appends items to the end of an array.
    
    Similar to Perl's push function. Appends one or more items to the end
    of the array and returns the new length of the array.
    
    Args:
        array: Array/list to modify
        *items: Items to append
        
    Returns:
        New length of the array (or None if array is invalid)
        
    Perl Source: Perl push(@array, items...)
    """
    if not isinstance(array, list):
        return None

    for item in items:
        if isinstance(item, (list, tuple)):
            array.extend(item)
        else:
            array.append(item)

    return len(array)


def pop(array: List[Any]) -> Any:
    """
    Pop function - removes and returns the last element from an array.
    
    Similar to Perl's pop function. Removes and returns the last element
    of the array.
    
    Args:
        array: Array/list to modify
        
    Returns:
        The last element of the array
        
    Perl Source: Perl pop(@array)
    """
    if not isinstance(array, list) or len(array) == 0:
        return None
    return array.pop()


def shift(array: List[Any]) -> Any:
    """
    Shift function - removes and returns the first element from an array.
    
    Similar to Perl's shift function. Removes and returns the first element
    of the array.
    
    Args:
        array: Array/list to modify
        
    Returns:
        The first element of the array
        
    Perl Source: Perl shift(@array)
    """
    if not isinstance(array, list) or len(array) == 0:
        return None
    return array.pop(0)


def unshift(array: List[Any], *items: Any) -> Optional[int]:
    """
    Unshift function - prepends items to the beginning of an array.
    
    Similar to Perl's unshift function. Prepends one or more items to the
    beginning of the array and returns the new length.
    
    Args:
        array: Array/list to modify
        *items: Items to prepend
        
    Returns:
        New length of the array (or None if array is invalid)
        
    Perl Source: Perl unshift(@array, items...)
    """
    if not isinstance(array, list):
        return None

    for i, item in enumerate(items):
        array.insert(i, item)

    return len(array)


def scalar(array: Any) -> int:
    """
    Scalar function - returns the length/size of an array or scalar context.
    
    In Perl, scalar() forces scalar context evaluation. For arrays, this 
    returns the number of elements. For other values, it returns 1 if the
    value is truthy, 0 if falsy.
    
    Args:
        array: Array/list or any value
        
    Returns:
        Length of array/list, or 1/0 for scalar values
        
    Perl Source: Perl scalar(@array) or scalar($value)
    """
    if isinstance(array, (list, tuple)):
        return len(array)
    elif hasattr(array, '__len__'):
        return len(array)
    else:
        # Scalar context for non-array values
        return 1 if array else 0


__all__ = [
    'splice',
    'push',
    'pop',
    'shift',
    'unshift',
    'scalar',
]

