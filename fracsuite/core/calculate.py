



from functools import partial
from multiprocessing import Pool
# from pathos.multiprocessing import ProcessingPool as Pool
from typing import Callable, TypeVar

T1 = TypeVar('T1')
T2 = TypeVar('T2')
def pooled(objects: list[T1],
           function: Callable[[T1], T2],
           advance = None,
           **kwargs) -> list[T2]:
    """
    Pool a function over a list of objects.
    Keyword arguments are passed to the function.

    Args:
        objects (list[T1]): Input objects.
        function (Callable[[T1], T2]): The function to call on every object.
        advance (Callable[[], Any], optional): Function that can be called to advance a progressbar. Defaults to None.

    Returns:
        list[T2]: Results created from function.
    """

    diag_func = partial(function, **kwargs)
    results: list[T2] = []
    with Pool() as pool:
        for result in pool.imap_unordered(diag_func, objects):
            results.append(result)
            if advance is not None:
                advance()
    return results