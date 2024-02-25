import inspect
from functools import wraps
from typing import Callable, get_type_hints

def mimic(original_func: Callable):
    def decorator(new_func: Callable):
        # Kopieren der Signatur von der originalen Funktion
        new_func.__signature__ = inspect.signature(original_func)
        # Kopieren der Typ-Hinweise von der originalen Funktion
        new_func.__annotations__ = get_type_hints(original_func)

        @wraps(original_func)
        def wrapper(*args, **kwargs):
            return new_func(*args, **kwargs)
        return wrapper
    return decorator
