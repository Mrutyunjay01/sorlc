import pytest
from chess_env.board import ChessBoard
from chess_env.rules import compute_reward, REWARD_WIN
from chess_env.renderer import print_board, render_move_history

def test_initial_state():
    board = ChessBoard()
    assert board.turn == "white"
    assert board.is_game_over is False
    assert len(board.legal_moves) == 20


def test_push_move():
    board = ChessBoard()
    result = board.push_move("e2e4")

    assert result.turn == "black"
    assert result.is_game_over is False
    assert "e7e5" in result.legal_moves


def test_illegal_move_raises():
    board = ChessBoard()

    with pytest.raises(ValueError):
        board.push_move("e2e5")


def test_material_balance():
    board = ChessBoard()
    assert board.material_balance == 0


def test_copy_independence():
    board = ChessBoard()
    board_copy = board.copy()

    board.push_move("e2e4")

    assert board_copy.turn == "white"
    assert len(board_copy.legal_moves) == 20


def test_renderer_runs_without_error():
    board = ChessBoard()
    print_board(board)  # just ensure no exception


def test_move_history_format():
    moves = ["e2e4", "e7e5", "g1f3", "b8c6"]
    history = render_move_history(moves)

    assert "1." in history
    assert "2." in history


def test_fool_mate():
    """
    Fool's mate: 1. f3 e5 2. g4 Qh4#
    """
    board = ChessBoard()

    moves = ["f2f3", "e7e5", "g2g4", "d8h4"]
    acting_colors = ["white", "black", "white", "black"]

    for i, (move, color) in enumerate(zip(moves, acting_colors)):
        board_state = board.push_move(move)

        if i == len(moves) - 1:
            reward = compute_reward(board_state, color)

            assert board_state.is_checkmate is True
            assert reward == REWARD_WIN