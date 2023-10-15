from typing import Callable


class Outputtable:

    def get_output_funcs(self) -> dict[str, Callable[[str], str]]:
        """Returns the output path functions for the object."""
        raise Exception(f"'get_output_path' is not implemented on {type(self)}")