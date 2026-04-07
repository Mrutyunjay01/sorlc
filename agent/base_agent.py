from abc import ABC, abstractmethod
from envs.base_env import BaseObservation, BaseAction

class BaseAgent(ABC):

    @abstractmethod
    def select_action(self, env_observation: BaseObservation) -> BaseAction:
        """ select a valid move from list of legal moves"""
        pass
    pass

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def on_transition(self, observation, action, step_result) -> None:
        """Optional hook for online-learning agents."""
        return None