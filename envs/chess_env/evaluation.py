import chess
from dataclasses import dataclass
from .board import ChessBoard

@dataclass
class EvalWeights:
    pawn: int = 100
    knight: int = 320
    bishop: int = 330
    rook: int = 500
    queen: int = 900
    pst_scale: float = 1.0
    mobility: float = 5.0
    center_control: float = 10.0


PAWN_TABLE = [
      0,   0,   0,   0,   0,   0,  0,   0,
      5,  10,  10, -20, -20,  10, 10,   5,
      5,  -5, -10,   0,   0, -10, -5,   5,
      0,   0,   0,  20,  20,   0,  0,   0,
      5,   5,  10,  25,  25,  10,  5,   5,
     10,  10,  20,  30,  30,  20, 10,  10,
     50,  50,  50,  50,  50,  50, 50,  50,
      0,   0,   0,   0,   0,   0,  0,   0,
]
KNIGHT_TABLE = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]
BISHOP_TABLE = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]
ROOK_TABLE = [
      0,   0,   5,  10,  10,   5,   0,   0,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
      5,  10,  10,  10,  10,  10,  10,   5,
      0,   0,   0,   0,   0,   0,   0,   0,
]
QUEEN_TABLE = [
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,   5,   5,   5,   0, -10,
     -5,   0,   5,   5,   5,   5,   0,  -5,
      0,   0,   5,   5,   5,   5,   0,  -5,
    -10,   5,   5,   5,   5,   5,   0, -10,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20,
]
KING_TABLE = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
     20,  20,   0,   0,   0,   0,  20,  20,
     20,  30,  10,   0,   0,  10,  30,  20,
]

PIECE_VALUES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
}

PIECE_SQUARE_TABLES = {
    chess.PAWN: PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK: ROOK_TABLE,
    chess.QUEEN: QUEEN_TABLE,
    chess.KING: KING_TABLE,
}

CENTER_SQUARES = [chess.D4, chess.E4, chess.D5, chess.E5]


def evaluate_board(board: ChessBoard, weights: EvalWeights | None = None) -> float:
    w = weights or EvalWeights()
    pyboard = board.python_board

    material = 0.0
    for piece_type, attr_name in PIECE_VALUES.items():
        val = float(getattr(w, attr_name))
        material += val * len(pyboard.pieces(piece_type, chess.WHITE))
        material -= val * len(pyboard.pieces(piece_type, chess.BLACK))

    pst_score = 0.0
    for piece_type, table in PIECE_SQUARE_TABLES.items():
        for square in pyboard.pieces(piece_type, chess.WHITE):
            pst_score += table[square]
        for square in pyboard.pieces(piece_type, chess.BLACK):
            pst_score -= table[chess.square_mirror(square)]
    pst_score *= w.pst_scale

    white_view = pyboard.copy(stack=False)
    white_view.turn = chess.WHITE
    black_view = pyboard.copy(stack=False)
    black_view.turn = chess.BLACK
    mobility_score = (white_view.legal_moves.count() - black_view.legal_moves.count()) * w.mobility

    center_score = 0.0
    for square in CENTER_SQUARES:
        center_score += len(pyboard.attackers(chess.WHITE, square))
        center_score -= len(pyboard.attackers(chess.BLACK, square))
    center_score *= w.center_control

    return material + pst_score + mobility_score + center_score
