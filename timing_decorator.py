import time
import functools


def measure_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"[TIME] {func.__name__} executed in {end - start:.4f} sec")
        return result
    return wrapper