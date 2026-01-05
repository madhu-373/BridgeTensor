"""
Generate C++ autograd code from derivatives.yaml.

This is the main code generator that creates backward function classes
and wrapper code for automatic differentiation.
"""

from load_derivatives import load_derivatives, DifferentiabilityInfo, Derivative
from typing import Dict, List
import os


class AutogradCodeGenerator:
    """Generates C++ autograd code from derivative specifications."""
    
    def __init__(self, infos: Dict[str, DifferentiabilityInfo]):
        self.infos = infos
        
    def generate_backward_class(self, info: DifferentiabilityInfo) -> str:
        """
        Generate a backward function class for one operation.
        
        For example, for 'add', generates:
        
        class AddBackward : public Node {
        public:
            AddBackward(TensorBase self, TensorBase other) 
                : self_(self), other_(other) {}
            
            std::vector<TensorBase> apply(std::vector<TensorBase>&& grads) override {
                auto grad = grads[0];
                return {grad, grad};
            }
        private:
            TensorBase self_;
            TensorBase other_;
        };
        """
        class_name = info.name.capitalize() + "Backward"
        
        # Generate constructor parameters (save inputs for backward)
        saved_inputs = []
        for deriv in info.derivatives:
            # Check if this input is needed in the formula
            formula = deriv.formula
            
            # Check if the input itself appears in its own gradient formula
            # e.g., for relu: self appears in "grad * (self > 0)"
            if deriv.var_name in formula and deriv.var_name not in saved_inputs:
                saved_inputs.append(deriv.var_name)
            
            # Check if OTHER inputs are referenced in this gradient
            for arg in info.args:
                # If formula references a different input variable, save it
                if arg != deriv.var_name and arg in formula:
                    if arg not in saved_inputs:
                        saved_inputs.append(arg)
        
        # Constructor
        ctor_params = []
        ctor_init = []
        members = []
        
        for arg in saved_inputs:
            ctor_params.append(f"TensorBase {arg}")
            ctor_init.append(f"{arg}_({arg})")
            members.append(f"    TensorBase {arg}_;")
        
        ctor_params_str = ", ".join(ctor_params)
        ctor_init_str = " : " + ", ".join(ctor_init) if ctor_init else ""
        
        # Constructor code
        if ctor_params:
            constructor = f"    {class_name}({ctor_params_str}){ctor_init_str} {{}}\n"
        else:
            constructor = f"    {class_name}() = default;\n"
        
        # Apply method - compute gradients
        apply_lines = []
        apply_lines.append("    std::vector<TensorBase> apply(std::vector<TensorBase>&& grads) override {")
        apply_lines.append("        auto grad = grads[0];")
        apply_lines.append("        return {")
        
        # Generate gradient for each input
        grad_exprs = []
        for deriv in info.derivatives:
            grad_expr = self.convert_formula(deriv.formula)
            grad_exprs.append(f"            {grad_expr}")
        
        apply_lines.append(",\n".join(grad_exprs))
        apply_lines.append("        };")
        apply_lines.append("    }")
        
        # Combine everything
        lines = []
        lines.append(f"/**")
        lines.append(f" * Backward function for {info.name}")
        lines.append(f" * Signature: {info.signature}")
        lines.append(f" */")
        lines.append(f"class {class_name} : public Node {{")
        lines.append("public:")
        lines.append(constructor)
        lines.append("")
        lines.extend(apply_lines)
        lines.append("")
        if members:
            lines.append("private:")
            lines.extend(members)
        lines.append("};")
        
        return "\n".join(lines)
    
    def convert_formula(self, formula: str) -> str:
        """
        Convert YAML derivative formula to C++ expression.
        
        Examples:
            "grad" -> "grad"
            "grad * other" -> "grad * other_"
            "grad.matmul(other.t())" -> "grad.matmul(other_.t())"
            "-grad" -> "-grad"
            "grad * (self > 0)" -> "grad * (self_ > 0)"
        """
        cpp_expr = formula
        
        # Replace variable references with saved member variables
        # Pattern: variable name followed by . or > or space or ) or ,
        # We need to be careful not to replace 'grad'
        
        # For simplicity, we'll replace specific patterns
        # Replace "self" with "self_" (but not in middle of words)
        cpp_expr = self._replace_var(cpp_expr, "self")
        cpp_expr = self._replace_var(cpp_expr, "other")
        
        return cpp_expr
    
    def _replace_var(self, expr: str, var: str) -> str:
        """Replace variable name with member variable (var -> var_)."""
        # Use word boundary replacement
        import re
        pattern = r'\b' + var + r'\b'
        return re.sub(pattern, var + '_', expr)
    
    def generate_header(self) -> str:
        """Generate the complete header file with all backward classes."""
        lines = []
        
        # Header guard and includes
        lines.append("#pragma once")
        lines.append("")
        lines.append("#include \"autograd/Node.h\"")
        lines.append("#include \"core/Tensor.h\"")
        lines.append("")
        lines.append("namespace OwnTensor {")
        lines.append("")
        lines.append("// Forward declaration")
        lines.append("class TensorBase;")
        lines.append("")
        lines.append("// ============================================================================")
        lines.append("// Generated Backward Functions")
        lines.append("// ============================================================================")
        lines.append("")
        
        # Generate each backward class
        for name in ["add", "matmul", "relu"]:  # Only these three
            if name in self.infos:
                lines.append(self.generate_backward_class(self.infos[name]))
                lines.append("")
        
        lines.append("} // namespace OwnTensor")
        
        return "\n".join(lines)
    
    def generate_wrapper_functions(self) -> str:
        """
        Generate wrapper functions that attach backward nodes.
        
        For 'add', generates:
        
        TensorBase add_autograd(const TensorBase& self, const TensorBase& other) {
            auto result = add(self, other);  // Call actual implementation
            
            if (/* any requires_grad */) {
                auto grad_fn = std::make_shared<AddBackward>(self, other);
                grad_fn->set_next_edge(0, make_edge(self.grad_fn(), 0));
                grad_fn->set_next_edge(1, make_edge(other.grad_fn(), 0));
                result.set_grad_fn(grad_fn);
            }
            
            return result;
        }
        """
        lines = []
        lines.append("// ============================================================================")
        lines.append("// Autograd Wrapper Functions")
        lines.append("// ============================================================================")
        lines.append("")
        
        for name in ["add", "matmul", "relu"]:
            if name in self.infos:
                lines.append(self.generate_wrapper_function(self.infos[name]))
                lines.append("")
        
        return "\n".join(lines)
    
    def generate_wrapper_function(self, info: DifferentiabilityInfo) -> str:
        """Generate a single wrapper function."""
        func_name = info.name
        class_name = func_name.capitalize() + "Backward"
        
        # Function signature
        params = []
        param_names = []
        for arg in info.args:
            params.append(f"const TensorBase& {arg}")
            param_names.append(arg)
        
        params_str = ", ".join(params)
        param_names_str = ", ".join(param_names)
        
        # Check which inputs need to be saved
        saved_inputs = []
        for deriv in info.derivatives:
            formula = deriv.formula
            for arg in info.args:
                if arg != deriv.var_name and arg in formula:
                    if arg not in saved_inputs:
                        saved_inputs.append(arg)
        
        saved_inputs_str = ", ".join(saved_inputs) if saved_inputs else ""
        
        lines = []
        lines.append(f"inline TensorBase {func_name}_autograd({params_str}) {{")
        lines.append(f"    auto result = {func_name}({param_names_str});")
        lines.append("")
        lines.append("    // Attach backward function if any input requires grad")
        lines.append(f"    // TODO: Check if any input requires_grad")
        lines.append(f"    auto grad_fn = std::make_shared<{class_name}>({saved_inputs_str});")
        lines.append("")
        lines.append("    // Set edges to input gradient functions")
        for i, arg in enumerate(info.args):
            lines.append(f"    // grad_fn->set_next_edge({i}, make_edge({arg}.grad_fn(), 0));")
        lines.append("")
        lines.append("    // result.set_grad_fn(grad_fn);")
        lines.append("    return result;")
        lines.append("}")
        
        return "\n".join(lines)


def main():
    """Main entry point for code generation."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(script_dir, "derivatives.yaml")
    
    # Load derivatives
    print("Loading derivatives.yaml...")
    infos = load_derivatives(yaml_path)
    
    # Filter to only add, matmul, relu
    filtered_infos = {k: v for k, v in infos.items() if k in ["add", "matmul", "relu"]}
    
    print(f"Generating code for: {list(filtered_infos.keys())}")
    
    # Generate code
    generator = AutogradCodeGenerator(filtered_infos)
    
    # Generate header
    header_code = generator.generate_header()
    
    # Write header file
    output_dir = os.path.join(script_dir, "../../include/autograd")
    os.makedirs(output_dir, exist_ok=True)
    
    header_path = os.path.join(output_dir, "Autograd_generated.h")
    with open(header_path, 'w') as f:
        f.write(header_code)
    
    print(f"Generated: {header_path}")
    
    # Generate wrapper functions (in comments for now)
    wrapper_code = generator.generate_wrapper_functions()
    print("\nGenerated wrapper functions (for reference):")
    print(wrapper_code)


if __name__ == "__main__":
    main()
