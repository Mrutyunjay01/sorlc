import chess
from dataclasses import dataclass
from typing import Optional

@dataclass
class BoardState:
    """ state of the board at any given point (mostly after a move) """
    fen: str                # fen notation of the board
    legal_moves: list[str]  # legal moves allowed given the state of the board
    is_game_over: bool
    is_checkmate: bool
    is_stalemate: bool
    is_draw: bool
    turn: str               # whose turn, black or white

class ChessBoard:
    """
    A chess game in UCI notation.
    """

    def __init__(self, fen: Optional[str] = None):
        """
        create a chess board from FEN (Forsyth Edwards Notation) -> represents a state of a game position
        for example: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        - first part is the board representation from black back rank to white.
        - then comes whose move it is, w or b
        - then comes castling availability, KQ means both King and Queen side castling available for white, lower case for black
        - "-" represents no enpassant move/square applicable
        - 0: number of moves since last pawn move (fo 50 move rule for draw)
        - 1: total move count (starts at 1)
        """
        self._board = chess.Board(fen=fen) if fen else chess.Board() # by default it opens up the starting position
        pass

    # some accessors/getters for utility
    @property
    def fen(self) -> str:
        """
        return complete board state as fen string
        """
        return self._board.fen()
    
    @property
    def turn(self) -> str:
        """ return whose turn it is """
        return "white" if self._board.turn == chess.WHITE else "black"
    
    @property
    def is_game_over(self) -> bool:
        return self._board.is_game_over()
    
    @property
    def is_stalemate(self) -> bool:
        return self._board.is_stalemate()
    
    @property
    def is_checkmate(self) -> bool:
        return self._board.is_checkmate()
    
    @property
    def is_draw(self) -> bool:
        return (
            self._board.is_stalemate() 
            or self._board.is_insufficient_material() 
            or self._board.is_fivefold_repetition() 
            or self._board.is_seventyfive_moves()
        )
    
    @property
    def legal_moves(self) -> list[str]:
        """ set of legal moves given board's state"""
        return [move.uci() for move in self._board.legal_moves]
    
    @property
    def material_balance(self) -> int:
        PIECE_VALUES = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900
        }
        balance = 0
        for piece_type, value in PIECE_VALUES.items():
            balance += value * len(self._board.pieces(piece_type, chess.WHITE))
            balance -= value * len(self._board.pieces(piece_type, chess.BLACK))
        
        return balance

    def push_move(self, uci_move: str) -> BoardState:
        """ make a move (in uci notation)"""

        if uci_move not in self.legal_moves:
            raise ValueError(
                f"Illegal move {uci_move} in position {self.fen} \n"
                f"Legal moves: {self.legal_moves}"
            )
        
        move = chess.Move.from_uci(uci=uci_move)
        
        self._board.push(move=move)

        # return something post move? may be a representational state of something
        return BoardState(
            fen=self.fen,
            legal_moves=self.legal_moves,
            is_stalemate=self.is_stalemate,
            is_game_over=self.is_game_over,
            is_checkmate=self.is_checkmate,
            is_draw=self.is_draw,
            turn=self.turn
        )

    def copy(self) -> "ChessBoard":
        """ return a deep-copy of the board (will be useful with agent-swarm or beam-search)"""
        new_board = ChessBoard()
        new_board._board = self._board.copy()
        return new_board
    pass