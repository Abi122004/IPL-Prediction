"""
Utility functions for the anonymization algorithm
"""

def cmp_str(s1, s2):
    """
    Compare two strings by their numerical value
    Args:
        s1: First string
        s2: Second string
    Returns:
        Difference between float values of s1 and s2
    """
    try:
        return float(s1) - float(s2)
    except ValueError:
        # If conversion fails, compare as strings
        return -1 if s1 < s2 else 1 if s1 > s2 else 0

def get_num_list_from_str(value):
    """
    Convert a string representation to a list of numbers
    Args:
        value: String representation (e.g., "1,5" or "10")
    Returns:
        List of strings representing numbers
    """
    if not value:
        return []
    try:
        if ',' in value:
            return value.split(',')
        else:
            return [value]
    except Exception:
        return [] 