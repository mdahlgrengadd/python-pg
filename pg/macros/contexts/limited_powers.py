"""
Limited Powers Context for WeBWorK.

This module provides context modifications that restrict the powers allowed
in polynomial expressions, useful for enforcing simplified form requirements.

Based on contextLimitedPowers.pl from the Perl WeBWorK distribution.
"""

from typing import Any, Optional


class LimitedPowers:
    """
    Context modifier to restrict polynomial powers.
    
    Provides methods to enforce limited power restrictions in mathematical
    expressions, such as disallowing only non-negative integers or only
    positive integers.
    """
    
    @staticmethod
    def OnlyIntegers(**kwargs: Any) -> None:
        """
        Restrict to integer powers only.
        
        Modifies the parsing context to accept only integer powers in
        polynomial expressions.
        
        Args:
            **kwargs: Configuration options (ignored in stub)
            
        Returns:
            None
            
        Perl Source: contextLimitedPowers.pl OnlyIntegers
        """
        pass
    
    @staticmethod
    def OnlyPositiveIntegers(**kwargs: Any) -> None:
        """
        Restrict to positive integer powers only.
        
        Modifies the parsing context to accept only positive integer powers
        in polynomial expressions.
        
        Args:
            **kwargs: Configuration options (ignored in stub)
            
        Returns:
            None
            
        Perl Source: contextLimitedPowers.pl OnlyPositiveIntegers
        """
        pass
    
    @staticmethod
    def OnlyNonNegativeIntegers(**kwargs: Any) -> None:
        """
        Restrict to non-negative integer powers only.
        
        Modifies the parsing context to accept only non-negative integer powers
        in polynomial expressions.
        
        Args:
            **kwargs: Configuration options (ignored in stub)
            
        Returns:
            None
            
        Perl Source: contextLimitedPowers.pl OnlyNonNegativeIntegers
        """
        pass


__all__ = [
    'LimitedPowers',
]

