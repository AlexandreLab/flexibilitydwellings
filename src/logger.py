import datetime as dt


def log_time_class(func):
    def wrapper(self, *args, **kwargs):
        time_start = dt.datetime.now()
        obj_result = func(self, *args, **kwargs)
        time_end = dt.datetime.now()
        print(
            f"{func.__name__} took {(time_end-time_start)} or {(time_end-time_start).microseconds}us"
        )
        return obj_result

    return wrapper
