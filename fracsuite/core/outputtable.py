from typing import Callable
import json
import numpy as np

class Outputtable:

    def get_output_funcs(self) -> dict[str, Callable[[str], str]]:
        """Returns the output path functions for the object."""
        raise Exception(f"'get_output_path' is not implemented on {type(self)}")


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)