from typing import Callable, Any

try:
    # prefer to import the actual njit callable if numba is available
    from numba import njit as _numba_njit  # type: ignore
    _NUMBA_AVAILABLE = True
except Exception:
    _numba_njit = None
    _NUMBA_AVAILABLE = False


def njit_wrapper(*dargs: Any, **dkwargs: Any) -> Callable:
    """
    A decorator that behaves like numba.njit when Numba is installed,
    and acts as a no-op decorator otherwise.

    Usage (both forms supported):
        @njit_wrapper
        def f(...): ...

        @njit_wrapper(parallel=True, nogil=True)
        def f(...): ...

    Implementation details:
    - If Numba is present:
        - `@njit_wrapper` (no args) returns the compiled function.
        - `@njit_wrapper(...)` returns `_numba_njit(...)` (a decorator) so the usual semantics apply.
    - If Numba is not present:
        - `@njit_wrapper` returns the original function.
        - `@njit_wrapper(...)` returns a no-op decorator that returns the original function.

    This helper intentionally mirrors the call patterns of numba.njit and is safe to use
    as a drop-in replacement in code that should work with or without Numba.
    """
    # Case: used as @njit_wrapper without parentheses — dargs[0] will be the function
    if _NUMBA_AVAILABLE:
        # Forward to real numba.njit. Need to handle two cases:
        # 1) @njit_wrapper  -> dargs contains (function,)
        # 2) @njit_wrapper(...) -> dargs empty and kwargs populated (or dargs used for positional args)
        if dargs and callable(dargs[0]) and not dkwargs:
            # Direct usage: @njit_wrapper above a function
            func = dargs[0]
            return _numba_njit(func)
        # Otherwise return a numba.njit decorator configured with provided args/kwargs
        return _numba_njit(*dargs, **dkwargs)
    else:
        # Numba not available: implement no-op decorator supporting both usages.
        if dargs and callable(dargs[0]) and not dkwargs:
            # Direct usage: @njit_wrapper above a function
            return dargs[0]
        # Otherwise return a decorator that returns the function unchanged
        def _noop_decorator(func: Callable) -> Callable:
            return func
        return _noop_decorator