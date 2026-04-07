from agent.base_agent import BaseAgent
from envs.chess_env import ChessAction, ChessObservation


class HumanAgent(BaseAgent):
    def __init__(self, name: str = "HumanAgent"):
        self._name = name
        self._move_provider = None

    def set_move_provider(self, provider) -> None:
        self._move_provider = provider

    def select_action(self, env_observation: ChessObservation) -> ChessAction:
        if env_observation.done:
            raise ValueError("cannot select action from terminal observation")
        if not env_observation.legal_moves:
            raise ValueError("cannot select action when no legal moves are available")

        if self._move_provider is not None:
            move = self._move_provider(env_observation)
            if move in env_observation.legal_moves:
                return ChessAction(move_uci=move)
            raise ValueError("UI selected an illegal move")

        print(f"\n{self._name} turn ({env_observation.turn}).")
        print(f"Legal moves: {', '.join(env_observation.legal_moves)}")

        while True:
            move = input("Enter move (UCI, e.g. e2e4): ").strip()
            if move in env_observation.legal_moves:
                return ChessAction(move_uci=move)
            print("Invalid move. Please enter one of the legal moves.")

    @property
    def name(self) -> str:
        return self._name
