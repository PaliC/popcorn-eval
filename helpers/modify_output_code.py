# Modifies the output_code.py you get from running torch.compile() with TORCH_LOGS="output_code"
# to save the PTX code of triton_* functions in a given output_directory
# Warning: This may be become stale eventually as it relies heavily on the current format of output_code.py which
# is traditionally used for debugging.

# example usage:
# run TORCH_LOGS="output_code" python compiled-model.py
# search "Output code written to:" string in the output. It should be followed by a file name.
# for each file name (oc.py), run 
# python modify_output_code.py --input_file oc.py --output_file modified_oc.py --output_dir triton_ptx
# python modified_oc.py 

import ast
import os

def rewrite_python_file(input_file_path, output_file_path, output_directory):
    with open(input_file_path, 'r') as f:
        source_code = f.read()
    
    # Parse the source code into an AST
    tree = ast.parse(source_code)
    base_name = os.path.basename(input_file_path)
    # remove .py extension
    base_name = base_name.replace('.py', '')
    
    class TritonCallTransformer(ast.NodeTransformer):
        def __init__(self, output_directory):
            self.output_directory = output_directory
            self.counter = 0
            self.need_import_os = False
            self.need_import_uuid = False

        def visit_FunctionDef(self, node):
            if node.name == 'call':
                # Visit the body of the 'call' function
                self.generic_visit(node)
            return node

        def visit_Expr(self, node):
            # Check if node.value is a function call
            if isinstance(node.value, ast.Call):
                call_node = node.value
                func = call_node.func
                # Check if function called is 'triton_*'
                if isinstance(func, ast.Attribute):
                    value = func.value
                    if isinstance(value, ast.Name):
                        func_name = value.id
                        if func_name.startswith('triton_'):
                            # Check if this call is not assigned
                            # Modify to assign to variable
                            self.counter += 1
                            var_name = f'triton_binary_{self.counter}'
                            assign_node = ast.Assign(
                                targets=[ast.Name(id=var_name, ctx=ast.Store())],
                                value=call_node
                            )
                            # Build code to save asm
                            save_asm_code = f"""
filename = os.path.join(r'{self.output_directory}', f'{base_name}_{func_name}' + '.ptx')
with open(filename, 'w') as f:
    f.write(triton_binary_{self.counter}.asm['ptx'])
"""
                            # Parse the save_asm_code into AST
                            save_asm_ast = ast.parse(save_asm_code).body
                            # Need to import os and uuid
                            self.need_import_os = True
                            self.need_import_uuid = True
                            # Replace the Expr node with assign_node and save_asm_ast
                            return [assign_node] + save_asm_ast
            return node

    # Apply the transformer
    transformer = TritonCallTransformer(output_directory)
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)

    # If needed, add import statements at the top
    if transformer.need_import_os or transformer.need_import_uuid:
        import_nodes = []
        if transformer.need_import_os:
            import_os_node = ast.Import(names=[ast.alias(name='os', asname=None)])
            import_nodes.append(import_os_node)
        if transformer.need_import_uuid:
            import_uuid_node = ast.Import(names=[ast.alias(name='uuid', asname=None)])
            import_nodes.append(import_uuid_node)
        # Insert import nodes after existing imports
        for idx, node in enumerate(new_tree.body):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                break
        new_tree.body = new_tree.body[:idx] + import_nodes + new_tree.body[idx:]

    # Write the modified code back to the file
    modified_code = ast.unparse(new_tree)
    with open(output_file_path, 'w') as f:
        f.write(modified_code)


# Example usage:
# Assuming 'code' contains the Python code as a string
# and 'output_dir' is the directory where you want to save the PTX files

# Read the content of the python file
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Modify Python file to extract Triton PTX code')
    parser.add_argument('--input_file', help='Input Python file to process')
    parser.add_argument('--output_file', help='Output modified Python file')
    parser.add_argument('--output-dir', required=True, help='Directory to save PTX files')
    
    args = parser.parse_args()

    rewrite_python_file(args.input_file, args.output_file, args.output_dir)

if __name__ == '__main__':
    main()
