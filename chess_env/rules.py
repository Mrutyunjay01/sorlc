from chess_env.board import ChessBoard, BoardState

REWARD_CHECKMATE =  100_000   # winning is the best possible outcome
REWARD_LOSS      = -100_000   # losing is the worst
REWARD_DRAW      =       0    # draw is neutral
REWARD_STEP      =       0    # non-terminal moves carry no reward


def compute_reward(board: ChessBoard, board_state: BoardState, acting_color: str) -> float:
    if not board_state.is_game_over:
        return REWARD_STEP

    if board_state.is_checkmate:
        # Reward is from white's perspective (consistent with evaluation sign).
        return REWARD_CHECKMATE if acting_color == "white" else REWARD_LOSS

    if board_state.is_stalemate:
        return REWARD_DRAW

    return REWARD_DRAW


def game_outcome(board: ChessBoard) -> str:
    if not board.is_game_over:
        return "Game in progress"
    if board.is_checkmate:
        winner = "Black" if board.turn == "white" else "White"
        return f"{winner} wins by checkmate"
    if board.is_stalemate:
        return "Draw by stalemate"
    return "Draw"