"""
Debug utilities for inspecting ProcessingElement graphs.

Copyright (c) 2026 R. Dunbar Poor, Andy Milburn and pygmu2 contributors

MIT License
"""

from typing import Set
from pygmu2.processing_element import ProcessingElement


def print_pe_tree(root: ProcessingElement, indent: int = 0, visited: Set[int] | None = None, current_path: Set[int] | None = None) -> None:
    """
    Recursively print a ProcessingElement tree/graph starting from the root.
    
    Walks the tree by following the inputs() method of each PE, printing
    each PE with indentation based on depth. Handles cycles by tracking
    nodes in the current path.
    
    Args:
        root: The root ProcessingElement to start from
        indent: Current indentation level (for recursive calls)
        visited: Set of object IDs already visited (to mark shared nodes)
        current_path: Set of object IDs in the current path (to detect cycles)
    
    Example:
        from pygmu2 import PortamentoPE, TransformPE, SinePE
        from pygmu2.debug_utils import print_pe_tree
        
        pitch_stream = PortamentoPE([(69, 0, 1000), (73, 1000, 1000)])
        freq_stream = TransformPE(pitch_stream, pitch_to_freq)
        synth_stream = SinePE(frequency=freq_stream)
        
        print_pe_tree(synth_stream)
    """
    if visited is None:
        visited = set()
    if current_path is None:
        current_path = set()
    
    # Use object ID to track nodes
    pe_id = id(root)
    
    # Check for cycles (node appears in current path)
    if pe_id in current_path:
        print("  " * indent + f"{root.__class__.__name__}: {root} [CYCLE DETECTED]")
        return
    
    # Check if we've seen this node before (shared node in DAG)
    is_shared = pe_id in visited
    if is_shared:
        print("  " * indent + f"{root.__class__.__name__}: {root} [shared]")
        return
    
    visited.add(pe_id)
    current_path.add(pe_id)
    
    # Print current PE with its repr
    print("  " * indent + f"{root.__class__.__name__}: {root}")
    
    # Get inputs and recurse
    inputs = root.inputs()
    if inputs:
        print("  " * indent + "  inputs:")
        for input_pe in inputs:
            print_pe_tree(input_pe, indent + 2, visited, current_path)
    
    # Remove from current path when backtracking
    current_path.remove(pe_id)
