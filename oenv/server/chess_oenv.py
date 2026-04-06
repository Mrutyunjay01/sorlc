from openenv.core import Environment
from chess_env.board import ChessBoard
from chess_env.rules import compute_reward, game_outcome
from oenv.model import ChessOpenEnvAction, ChessOpenEnvObservation, ChessOpenEnvState


class ChessOpenEnv(Environment[ChessOpenEnvAction, ChessOpenEnvObservation, ChessOpenEnvState]):
    def __init__(
            self, 
            fen: str = None,
            transform = None, 
            rubric = None
        ):
        # create a chess environment
        self._board = ChessBoard(fen=fen)
        self._move_count = 0
        self._move_limit = 200
    
    def reset(
            self, 
            fen: str = None,
            seed = None, 
            episode_id = None, 
            **kwargs
        ) -> ChessOpenEnvObservation:
        if not self._board:
            raise ValueError("no active ChessEnv found, create one first")
        
        self._board = ChessBoard(fen=fen)
        self._move_count = 0
        return self._observe(
            reward_value=0.0,
            done=False,
            meta_info={
                "outcome": "in progress",
                "move_count": self._move_count,
                "acting_color": None,
            },
        )
    
    @property
    def state(self) -> ChessOpenEnvState:
        return ChessOpenEnvState(
            episode_id=getattr(self, "_episode_id", None),
            step_count=self._move_count,
            fen=self._board.fen,
            turn=self._board.turn,
        )
    
    def step(
            self, 
            action: ChessOpenEnvAction, 
            timeout_s = None, 
            **kwargs
        ) -> ChessOpenEnvObservation:
        """Apply action, advance the world, return the result."""
        if self._board is None:
            raise RuntimeError("Call reset() before step().")
        
        acting_color = self._board.turn
        board_state = self._board.push_move(action.move_uci)
        self._move_count += 1
 
        reward_value = compute_reward(self._board, board_state, acting_color)
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
        return observation
    
    def _observe(self, reward_value, done: bool, meta_info: dict) -> ChessOpenEnvObservation:
        return ChessOpenEnvObservation(
            fen=self._board.fen,
            reward=reward_value,
            legal_moves=self._board.legal_moves,
            evaluation=self._board.material_balance,   # centipawns, not material_balance
            turn=self._board.turn,
            done=done,
            meta_info=meta_info
        )