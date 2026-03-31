from dataclasses import dataclass
from abc import ABC, abstractmethod

class BaseAction(ABC):
    pass

class BaseObservation(ABC):
    pass

class BaseReward(ABC):
    pass

@dataclass
class BaseStepResult(ABC):
    observation: BaseObservation
    reward: BaseReward
    is_terminal: bool
    meta_info: dict


class BaseEnv(ABC):

    @abstractmethod
    def step(self, action: BaseAction, **kwargs) -> BaseStepResult:
        pass

    @abstractmethod
    def reset(self, **kwargs):
        pass

    @abstractmethod
    def get_state(self) -> BaseObservation:
        """ get state of the environment (align with open-env specs) """
        ...
