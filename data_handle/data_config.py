"""
Backward Compatibility Module: case_table

This module has been migrated to data_handle._case_table.
Import redirection is kept here to avoid breaking existing code.

New code should use:
    from data_handle import case_table
"""

from data_handle._case_table import case_table

__all__ = ['case_table']
