from time import time


class Timer:
    def __init__(self) -> None:
        self.last_time = None

    def get_rate(self, n: int) -> str:
        current = time()
        if self.last_time is None:
            self.last_time = current
            return "No rate."
        rate = n / (current - self.last_time)
        self.last_time = current
        return f"{rate:.3f} it/s"
