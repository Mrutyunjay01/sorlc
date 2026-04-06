from random import Random
from agent.base_agent import BaseAgent
from envs.chess_env import ChessObservation, ChessAction

class RandomAgent(BaseAgent):
    def __init__(self, seed: int = None):
        self._range = Random(seed)

    def select_action(self, env_observation: ChessObservation) -> ChessAction:
        if env_observation.done:
            raise ValueError("cannot select action from terminal observation")
        if not env_observation.legal_moves:
            raise ValueError("cannot select action when no legal moves are available")
        move = self._range.choice(env_observation.legal_moves) # just randomly pick a move from the legal moves
        return ChessAction(move_uci=move)