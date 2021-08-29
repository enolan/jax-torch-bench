from typing import Optional


class ConfigurationError(Exception):
    def __init__(self, err):
        self.err = err


class ShapeError(Exception):
    def __init(self, err):
        self.err = err


class EWMA:
    def __init__(self, smoothing_factor: float):
        self.alpha = 1 - smoothing_factor
        self.average: Optional[float] = None

    def update_ewma(self, new_value: float) -> float:
        if self.average is not None:
            self.average = self.alpha * new_value + (1 - self.alpha) * self.average
        else:
            self.average = new_value
        return self.average
