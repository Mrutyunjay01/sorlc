from dataclasses import dataclass
from chess_env.board import ChessBoard
from chess_env.evaluation import evaluate_board
from chess_env.rules import compute_reward, game_outcome
from envs.base_env import BaseEnv, BaseAction, BaseObservation, BaseState, BaseStepResult

@dataclass
class ChessAction(BaseAction):
    move_uci: str

@dataclass
class ChessObservation(BaseObservation):
    """ represents an observation (not a state) from the env """
    fen         : str
    legal_moves : list[str]
    evaluation  : int
    turn        : str
    pass

@dataclass
class ChessState(BaseState):
    step_count: int
    fen: str
    turn: str

@dataclass
class ChessStepResult(BaseStepResult):
    ...

class ChessEnv(BaseEnv):
    def __init__(self, fen: str = None):
        # create a chess environment
        self._board = ChessBoard(fen=fen)
        self._move_count = 0
        self._move_limit = 200

    def reset(self, fen: str = None, **kwargs) -> ChessStepResult:
        if not self._board:
            raise ValueError("no active ChessEnv found, create one first")
        
        self._board = ChessBoard(fen=fen)
        self._move_count = 0
        observation = self._observe(
            reward_value=0.0,
            done=False,
            meta_info={
                "outcome": "in progress",
                "move_count": self._move_count,
                "acting_color": None,
            },
        )
        return ChessStepResult(
            observation=observation,
            reward=0.0,
            done=False
        )
    
    def step(self, action: ChessAction, **kwargs) -> BaseStepResult:
        """Apply action, advance the world, return the result."""
        if self._board is None:
            raise RuntimeError("Call reset() before step().")
 
        acting_color = self._board.turn
        previous_evaluation = evaluate_board(self._board)
        board_state = self._board.push_move(action.move_uci)
        self._move_count += 1
 
        reward_value = compute_reward(
            self._board,
            board_state,
            acting_color,
            previous_evaluation=previous_evaluation,
        )
        is_terminal = board_state.is_game_over or self._move_count >= self._move_limit
        if is_terminal and not board_state.is_game_over:
            outcome = "Draw by move limit"
        else:
            outcome = game_outcome(self._board) if is_terminal else "in progress"
        meta_info={
                "outcome": outcome,
                "move_count": self._move_count,
                "acting_color": acting_color,
            }

        observation = self._observe(reward_value=reward_value, done=is_terminal, meta_info=meta_info)
        return ChessStepResult(
            observation=observation,
            reward=reward_value,
            done=is_terminal
        )
    
    @property
    def state(self):
        return ChessState(
            step_count=self._move_count,
            fen=self._board.fen,
            turn=self._board.turn,
        )
    
    def _observe(self, reward_value, done: bool, meta_info: dict | None) -> ChessObservation:
        return ChessObservation(
            fen=self._board.fen,
            reward=reward_value,
            legal_moves=self._board.legal_moves,
            evaluation=evaluate_board(self._board),
            turn=self._board.turn,
            done=done,
            meta_info=meta_info
        )