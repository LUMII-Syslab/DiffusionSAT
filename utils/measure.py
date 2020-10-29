import time


class Timer:

    def __init__(self, start) -> None:
        self.__start_time = time.time() if start else None

    def start(self):
        self.__start_time = time.time()

    def lap_time(self):
        lap = time.time() - self.__start_time
        self.__start_time = time.time()
        return lap
