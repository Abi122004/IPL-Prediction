"""
NumRange module for numerical attributes
"""

class NumRange:
    """
    Class for numerical attributes
    """
    def __init__(self, sort_value, value):
        """
        Initialize with sorted values and range string
        sort_value: list of all possible values in sorted order
        value: string representation of range (e.g., "0,99")
        """
        self.sort_value = sort_value
        self.value = value 