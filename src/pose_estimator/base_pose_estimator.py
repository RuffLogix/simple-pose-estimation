from abc import ABC, abstractmethod


class BaseEstimator(ABC):
    def __init__(self, model_path: str):
        self.model_path = model_path

    @abstractmethod
    def estimate_pose(self, *args, **kwargs) -> tuple:
        pass
