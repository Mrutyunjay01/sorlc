import pytest
from agent.minmax_agent import MinimaxAgent
from agent.random_agent import RandomAgent
from agent.human_agent import HumanAgent
from chess_env.evaluation import evaluate_board
from envs.chess_env import ChessAction, ChessEnv


def test_step_propagates_reward_and_observation_reward():
    env = ChessEnv()
    step = env.reset()
    action = ChessAction(move_uci=step.observation.legal_moves[0])
    step = env.step(action)

    assert step.reward == step.observation.reward
    assert step.done == step.observation.done


def test_non_terminal_reward_is_eval_delta():
    env = ChessEnv()
    step = env.reset()
    prev_eval = evaluate_board(env._board)
    action = ChessAction(move_uci="e2e4")
    step = env.step(action)
    next_eval = evaluate_board(env._board)
    assert step.done is False
    assert step.reward == pytest.approx(next_eval - prev_eval, rel=1e-9, abs=1e-9)


def test_move_limit_terminal_has_terminal_outcome():
    env = ChessEnv()
    env._move_limit = 1
    step = env.reset()
    action = ChessAction(move_uci=step.observation.legal_moves[0])
    step = env.step(action)

    assert step.done is True
    assert step.observation.meta_info["outcome"] != "in progress"


def test_reset_and_step_meta_shape_is_consistent():
    env = ChessEnv()
    reset_step = env.reset()
    step = env.step(ChessAction(move_uci=reset_step.observation.legal_moves[0]))

    reset_keys = set(reset_step.observation.meta_info.keys())
    step_keys = set(step.observation.meta_info.keys())
    assert reset_keys == step_keys


def test_agents_raise_on_terminal_observation():
    terminal_fen = "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"
    env = ChessEnv(fen=terminal_fen)
    reset_step = env.reset(fen=terminal_fen)
    obs = reset_step.observation

    assert obs.done is False
    assert obs.legal_moves == []

    with pytest.raises(ValueError):
        RandomAgent(seed=0).select_action(obs)

    with pytest.raises(ValueError):
        MinimaxAgent(depth=2).select_action(obs)


def test_minimax_finds_forced_mate_in_one():
    # Position after: 1.f3 e5 2.g4 ; black to move has Qh4#.
    fen = "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq g3 0 2"
    env = ChessEnv(fen=fen)
    obs = env.reset(fen=fen).observation

    move = MinimaxAgent(depth=1).select_action(obs)
    assert move.move_uci == "d8h4"


def test_minimax_uses_reward_delta_signal():
    env = ChessEnv()
    obs = env.reset().observation
    move = MinimaxAgent(depth=1).select_action(obs)
    assert move.move_uci == "e2e4"


def test_local_openenv_semantics_are_aligned_without_server():
    openenv_core = pytest.importorskip("openenv.core")
    assert openenv_core is not None

    from oenv.server.chess_oenv import ChessOpenEnv

    local_env = ChessEnv()
    local_reset = local_env.reset()
    action = ChessAction(move_uci=local_reset.observation.legal_moves[0])
    local_step = local_env.step(action)

    oenv = ChessOpenEnv()
    _ = oenv.reset()
    oenv_step_obs = oenv.step(action)

    assert local_step.observation.fen == oenv_step_obs.fen
    assert local_step.observation.turn == oenv_step_obs.turn
    assert local_step.observation.legal_moves == oenv_step_obs.legal_moves
    assert local_step.reward == oenv_step_obs.reward
    assert local_step.done == oenv_step_obs.done
    assert set(local_step.observation.meta_info.keys()) == set(oenv_step_obs.meta_info.keys())


def test_openenv_client_parse_result_keeps_reward_done_evaluation():
    pytest.importorskip("openenv.core")
    from oenv.client.chess_oenv_client import ChessOpenEnvClient

    client = ChessOpenEnvClient(base_url="http://localhost:8000")
    payload = {
        "observation": {
            "fen": "test-fen",
            "legal_moves": ["e2e4"],
            "evaluation": 12,
            "turn": "white",
            "meta_info": {"outcome": "in progress"},
        },
        "reward": 3.5,
        "done": False,
    }

    step_result = client._parse_result(payload)
    assert step_result.reward == 3.5
    assert step_result.done is False
    assert step_result.observation.evaluation == 12


def test_human_agent_name_property():
    agent = HumanAgent()
    assert "Human" in agent.name
