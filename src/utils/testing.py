'''
Utils for testing and debugging.
'''

import copy
from functools import wraps
import time
import typing as tp


VERIFY_NOT_MODIFIED = False


def verify_not_modified(act: bool) -> None:
    def wrapper_wrapper(func: tp.Callable) -> tp.Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if act:
                args_copy = copy.deepcopy(args)
                kwargs_copy = copy.deepcopy(kwargs)

            result = func(*args, **kwargs)

            if act:
                assert str(args) == str(args_copy), f"{args=}\n{args_copy=}"
                assert str(kwargs) == str(kwargs_copy), f"{kwargs=}\n{kwargs_copy=}"

            return result
        return wrapper
    return wrapper_wrapper


@verify_not_modified(VERIFY_NOT_MODIFIED)
def timer(func: tp.Callable, *args, **kwargs) -> tuple[tp.Any, float]:
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time


_T = tp.TypeVar('_T')
_P = tp.ParamSpec('_P')
def time_estimator(func: tp.Callable[_P, _T]) -> tp.Callable[_P, _T]:
    @wraps(func)
    def wrapper(*args, **kwargs) -> tuple[tp.Any, float]:
        return timer(func, *args, **kwargs)
    return wrapper
