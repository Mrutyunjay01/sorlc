from dataclasses import dataclass
from chess_env.board import ChessBoard
from chess_env.rules import compute_reward, game_outcome
from envs.base_env import BaseEnv, BaseAction, BaseObservation, BaseStepResult, BaseReward

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
class ChessReward(BaseReward):
    _value: float

    @property
    def value(self) -> float:
        return self._value

@dataclass
class ChessStepResult(BaseStepResult):
    ...

class ChessEnv(BaseEnv):
    def __init__(self, fen: str = None):
        # create a chess environment
        self._board = ChessBoard(fen=fen)
        self._move_count = 0
        self._move_limit = 200

    def reset(self, fen: str = None, **kwargs):
        if not self._board:
            raise ValueError("no active ChessEnv found, create one first")
        
        self._board = ChessBoard(fen=fen)
        self._move_count = 0
        return self._observe()
    
    def step(self, action: ChessAction, **kwargs) -> ChessStepResult:
        """Apply action, advance the world, return the result."""
        if self._board is None:
            raise RuntimeError("Call reset() before step().")
 
        acting_color = self._board.turn
        board_state = self._board.push_move(action.move_uci)
        self._move_count += 1
 
        reward_value = compute_reward(self._board, board_state, acting_color)
        is_terminal = board_state.is_game_over or self._move_count >= self._move_limit
 
        return ChessStepResult(
            observation=self._observe(),
            reward=ChessReward(_value=reward_value),
            is_terminal=is_terminal,
            meta_info={
                "outcome": game_outcome(self._board) if is_terminal else "in progress",
                "move_count": self._move_count,
                "acting_color": acting_color,
            }
        )
    
    def get_state(self) -> ChessObservation:
        return self._observe()
    
    def _observe(self) -> ChessObservation:
        return ChessObservation(
            fen=self._board.fen,
            legal_moves=self._board.legal_moves,
            evaluation=self._board.material_balance,   # centipawns, not material_balance
            turn=self._board.turn,
        )