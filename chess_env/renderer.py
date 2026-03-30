from .board import ChessBoard
from .rules import game_outcome

UNICODE_PIECES = {
    'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',  # White
    'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟',  # Black
}


def render_board(board: ChessBoard, use_unicode: bool = True) -> str:

    fen_board = board.fen.split(" ")[0]
    ranks = fen_board.split("/")   # 8 ranks, rank 8 first

    lines = []
    lines.append("  ┌─────────────────┐")

    for rank_idx, rank_str in enumerate(ranks):
        rank_number = 8 - rank_idx   # Rank 8 at top, rank 1 at bottom
        row_squares = _expand_rank(rank_str)

        cells = []
        for sq_idx, piece_char in enumerate(row_squares):
            if piece_char == ".":
                # Empty square — use a light/dark pattern for readability
                is_light = (rank_idx + sq_idx) % 2 == 0
                cells.append("·" if is_light else " ")
            else:
                symbol = UNICODE_PIECES.get(piece_char, piece_char) if use_unicode else piece_char
                cells.append(symbol)

        lines.append(f"{rank_number} │ {' '.join(cells)} │")

    lines.append("  └─────────────────┘")
    lines.append("    a b c d e f g h")

    return "\n".join(lines)


def render_status(board: ChessBoard) -> str:
    if board.is_game_over:
        return f"  Game over: {game_outcome(board)}"

    material = board.material_balance
    material_str = (
        f"  Material: White +{material}" if material > 0 else
        f"  Material: Black +{abs(material)}" if material < 0 else
        "  Material: Equal"
    )
    return f"  Turn: {board.turn.capitalize()}   {material_str}"


def print_board(board: ChessBoard, last_move: str = None) -> None:
    if last_move:
        print(f"\n  Last move: {last_move}")
    print(render_board(board))
    print(render_status(board))
    print()


def render_move_history(moves: list[str]) -> str:
    lines = []
    for i in range(0, len(moves), 2):
        move_num = i // 2 + 1
        white_move = moves[i]
        black_move = moves[i + 1] if i + 1 < len(moves) else "..."
        lines.append(f"  {move_num:>3}. {white_move:<8} {black_move}")
    return "\n".join(lines)


def _expand_rank(rank_str: str) -> list[str]:
    
    squares = []
    for char in rank_str:
        if char.isdigit():
            squares.extend(["."] * int(char))
        else:
            squares.append(char)
    return squares