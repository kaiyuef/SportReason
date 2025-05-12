# runtime_module.py
import types
import sys

class RuntimeModule:
    @staticmethod
    def from_string(module_name: str, file_path: str, source_code: str):
        """
        Dynamically create a Python module from a string of code.

        Args:
            module_name (str): Name to assign the created module.
            file_path (str): Ignored (kept for compatibility).
            source_code (str): The Python source code to execute.

        Returns:
            module: A live module object with executed code in its namespace.
        """
        module = types.ModuleType(module_name)
        sys.modules[module_name] = module
        exec(source_code, module.__dict__)
        return module
