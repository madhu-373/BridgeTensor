"""
Load and parse derivatives.yaml file.

This module loads derivative formulas from derivatives.yaml and creates
DifferentiabilityInfo objects that describe how to compute gradients.
"""

import yaml
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
import re


@dataclass
class Derivative:
    """Represents a derivative formula for one input argument."""
    var_name: str  # e.g., "self", "other"
    formula: str   # e.g., "grad", "grad * other"
    
    def __repr__(self):
        return f"Derivative({self.var_name}: {self.formula})"


@dataclass
class DifferentiabilityInfo:
    """Information about how to differentiate a function."""
    name: str  # Function name, e.g., "add", "matmul"
    signature: str  # Full signature, e.g., "add(Tensor self, Tensor other) -> Tensor"
    args: List[str]  # Input argument names, e.g., ["self", "other"]
    return_type: str  # Return type, e.g., "Tensor"
    derivatives: List[Derivative]  # Derivative for each input
    
    def __repr__(self):
        return f"DifferentiabilityInfo({self.name}, derivatives={self.derivatives})"


def parse_signature(sig: str) -> tuple[str, List[str], str]:
    """
    Parse function signature into components.
    
    Args:
        sig: e.g., "add(Tensor self, Tensor other) -> Tensor"
        
    Returns:
        (name, args, return_type)
        e.g., ("add", ["self", "other"], "Tensor")
    """
    # Match: name(args) -> return_type
    match = re.match(r'(\w+)\((.*?)\)\s*->\s*(\w+)', sig)
    if not match:
        raise ValueError(f"Invalid signature: {sig}")
    
    func_name = match.group(1)
    args_str = match.group(2).strip()
    return_type = match.group(3)
    
    # Parse arguments: "Tensor self, Tensor other" -> ["self", "other"]
    args = []
    if args_str:
        for arg in args_str.split(','):
            arg = arg.strip()
            # Extract just the variable name (last word)
            parts = arg.split()
            if parts:
                args.append(parts[-1])
    
    return func_name, args, return_type


def load_derivatives(derivatives_yaml_path: str) -> Dict[str, DifferentiabilityInfo]:
    """
    Load derivatives from YAML file.
    
    Args:
        derivatives_yaml_path: Path to derivatives.yaml
        
    Returns:
        Dictionary mapping function name to DifferentiabilityInfo
    """
    with open(derivatives_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    infos = {}
    
    for entry in data:
        if entry is None:
            continue
            
        # Get signature
        signature = entry.get('name')
        if not signature:
            continue
        
        # Parse signature
        func_name, args, return_type = parse_signature(signature)
        
        # Parse derivatives for each argument
        derivatives = []
        for arg in args:
            if arg in entry:
                formula = entry[arg]
                derivatives.append(Derivative(var_name=arg, formula=formula))
        
        # Create DifferentiabilityInfo
        info = DifferentiabilityInfo(
            name=func_name,
            signature=signature,
            args=args,
            return_type=return_type,
            derivatives=derivatives
        )
        
        infos[func_name] = info
    
    return infos


if __name__ == "__main__":
    # Test the parser
    import sys
    import os
    
    yaml_path = os.path.join(os.path.dirname(__file__), "derivatives.yaml")
    infos = load_derivatives(yaml_path)
    
    print("Loaded derivatives:")
    for name, info in infos.items():
        print(f"\n{name}:")
        print(f"  Signature: {info.signature}")
        print(f"  Args: {info.args}")
        print(f"  Derivatives:")
        for deriv in info.derivatives:
            print(f"    {deriv.var_name}: {deriv.formula}")
