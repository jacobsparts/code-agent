"""
Source extraction utilities for injecting methods into subprocesses.

Used by REPLAgent for inject=True tools.
"""

import ast
import inspect
import textwrap
from typing import Callable


def extract_method_source(impl: Callable, name: str) -> str:
    """
    Extract and transform method source for injection into subprocess.
    
    Transforms a bound method into a standalone function by:
    - Removing 'self' parameter
    - Stripping type annotations (may reference unavailable types)
    - Converting string defaults to None (Code Agent convention)
    - Removing decorators
    - Replacing getattr(self, 'attr', default) with default
    
    Args:
        impl: Method to extract
        name: Function name (for error messages)
        
    Returns:
        Python source code for the standalone function
        
    Raises:
        ValueError: If source extraction or transformation fails
    """
    # Prefer the source captured when the tool was decorated.  Calling
    # inspect.getsource() later can combine old function line metadata with a
    # changed on-disk source file and inject the wrong function body.
    source = getattr(impl, "_tool_source", None)
    if source is None:
        try:
            source = inspect.getsource(impl)
        except (OSError, TypeError) as e:
            raise ValueError(f"Cannot extract source for '{name}': {e}")
    
    source = textwrap.dedent(source)
    
    # Parse to AST
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise ValueError(f"Cannot parse source for '{name}': {e}")
    
    # Find the top-level function definition.  Do not use ast.walk() here:
    # injected tools may contain nested helper functions/classes, and extracting
    # one of those would silently replace the tool with the wrong callable.
    top_level_functions = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    func_def = None
    for node in top_level_functions:
        if node.name == name:
            func_def = node
            break

    if func_def is None and len(top_level_functions) == 1:
        func_def = top_level_functions[0]
        func_def.name = name

    if func_def is None:
        raise ValueError(f"Cannot find function '{name}' in extracted source")
    
    # Remove 'self' parameter
    if func_def.args.args and func_def.args.args[0].arg == 'self':
        func_def.args.args.pop(0)
    
    # Strip type annotations
    for arg in func_def.args.args:
        arg.annotation = None
    for arg in func_def.args.kwonlyargs:
        arg.annotation = None
    if func_def.args.vararg:
        func_def.args.vararg.annotation = None
    if func_def.args.kwarg:
        func_def.args.kwarg.annotation = None
    func_def.returns = None
    
    # Fix string defaults (Code Agent convention: strings are descriptions, not values)
    new_defaults = []
    for default in func_def.args.defaults:
        if isinstance(default, ast.Constant) and isinstance(default.value, str):
            new_defaults.append(ast.Constant(value=None))
        else:
            new_defaults.append(default)
    func_def.args.defaults = new_defaults
    
    # Remove decorators
    func_def.decorator_list = []
    
    # Replace getattr(self, 'attr', default) with just default
    class SelfGetattrReplacer(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if (isinstance(node.func, ast.Name) and
                node.func.id == 'getattr' and
                len(node.args) >= 3 and
                isinstance(node.args[0], ast.Name) and
                node.args[0].id == 'self'):
                return node.args[2]
            return node
    
    func_def = SelfGetattrReplacer().visit(func_def)
    ast.fix_missing_locations(func_def)
    
    return ast.unparse(func_def)
