



from multiprocessing import Pool
from typing import Callable, TypeVar

T1 = TypeVar('T1')
T2 = TypeVar('T2')
def pooled(objects: list[T1],
           function: Callable[[T1], T2],
           advance = None) -> list[T2]:
    results: list[T2] = []
    with Pool() as pool:
        for result in pool.imap_unordered(function, objects):
            results.append(result)
            if advance is not None:
                advance()
    return results