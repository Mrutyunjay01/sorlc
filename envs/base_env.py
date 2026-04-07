from dataclasses import dataclass
from abc import ABC, abstractmethod

class BaseAction(ABC):
    pass

@dataclass
class BaseObservation(ABC):
    reward: int | float | None
    done: bool
    meta_info: dict
    pass

@dataclass
class BaseStepResult(ABC):
    observation: BaseObservation
    reward:      int | float
    done:        bool

class BaseState(ABC):
    step_count: int

class BaseEnv(ABC):

    @abstractmethod
    def step(self, action: BaseAction, **kwargs) -> BaseObservation:
        pass

    @abstractmethod
    def reset(self, **kwargs):
        pass

    @property
    @abstractmethod
    def state(self) -> BaseState:
        """ get state of the environment (align with open-env specs) """
        ...
