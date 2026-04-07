import time
import asyncio
import argparse
from agent.base_agent import BaseAgent
from agent.random_agent import RandomAgent
from agent.minmax_agent import MinimaxAgent
from agent.human_agent import HumanAgent
from oenv.client.chess_oenv_client import ChessOpenEnvClient


def _build_ui(enabled: bool):
    if not enabled:
        return None
    try:
        from ui.chess_tk import ChessTkUI
        return ChessTkUI(title="OpenEnv Chess Client")
    except Exception as exc:
        print(f"UI unavailable ({exc}), continuing without UI.")
        return None

async def play_game(
    white: BaseAgent,
    black: BaseAgent,
    verbose: bool = True,
    ui_enabled: bool = False,
) -> dict:
    async with ChessOpenEnvClient(base_url="http://localhost:8000") as env:
        reset_step_result = await env.reset() # output of reset is also a step result
        obs = reset_step_result.observation
        moves = []
        start = time.time()
        ui = _build_ui(ui_enabled)
        if ui:
            ui.render(obs.fen, status_text=f"Turn: {obs.turn}")

        if verbose:
            print(f"\n{'='*50}")
            print(f"  {white.name} (White)  vs  {black.name} (Black)")
            print(f"{'='*50}")

        while True:
            agent = white if obs.turn == "white" else black
            previous_obs = obs
            if ui and isinstance(agent, HumanAgent):
                agent.set_move_provider(ui.prompt_move)
            action = agent.select_action(previous_obs)
            step_result = await env.step(action=action)
            agent.on_transition(previous_obs, action, step_result)
            moves.append(action.move_uci)
            obs = step_result.observation
            if ui:
                ui.render(
                    obs.fen,
                    status_text=f"Turn: {obs.turn} | Eval: {obs.evaluation:.1f} | Reward: {step_result.reward:.1f}",
                    last_move=action.move_uci,
                )

            if step_result.done:
                break

        duration = time.time() - start
        outcome_str = obs.meta_info["outcome"]
        winner = (
            "white" if "White wins" in outcome_str else
            "black" if "Black wins" in outcome_str else
            "draw"
        )

        if verbose:
            print(f"  Result: {outcome_str}")
            print(f"  Moves:  {len(moves)}")
            print(f"  Time:   {duration:.1f}s")
        if ui:
            ui.render(
                obs.fen,
                status_text=f"Game Over: {outcome_str}",
                last_move=moves[-1] if moves else None,
            )
            await asyncio.sleep(1.5)
            ui.close()

        return {
            "outcome": winner,
            "outcome_str": outcome_str,
            "move_count": len(moves),
            "moves": moves,
            "duration_s": duration,
        }


async def run_match(
    white: BaseAgent,
    black: BaseAgent,
    n_games: int = 1,
    verbose: bool = True,
    ui_enabled: bool = False,
):
    tally = {"white": 0, "black": 0, "draw": 0}
    total_moves = 0
    total_time = 0.0
    per_game_verbose = verbose and n_games == 1

    for game_num in range(1, n_games + 1):
        if n_games > 1:
            print(f"  Game {game_num}/{n_games}...", end="\n", flush=True)

        result = await play_game(white, black, verbose=per_game_verbose, ui_enabled=ui_enabled)
        tally[result["outcome"]] += 1
        total_moves += result["move_count"]
        total_time += result["duration_s"]

        if n_games > 1:
            print(f"{result['outcome_str']}  ({result['move_count']} moves, {result['duration_s']:.1f}s)")

    if n_games > 1:
        print(f"\n{'='*50}")
        print(f"  Match summary: {n_games} games")
        print(f"{'='*50}")
        print(f"  {white.name} (White) wins:  {tally['white']}  ({100*tally['white']/n_games:.0f}%)")
        print(f"  {black.name} (Black) wins:  {tally['black']}  ({100*tally['black']/n_games:.0f}%)")
        print(f"  Draws:                    {tally['draw']}  ({100*tally['draw']/n_games:.0f}%)")
        print(f"  Avg moves per game:       {total_moves/n_games:.1f}")
        print(f"  Total time:               {total_time:.1f}s")


def build_agent(name: str, depth: int, seed: int = None) -> BaseAgent:
    if name == "random":
        return RandomAgent(seed=seed)
    elif name == "minimax":
        return MinimaxAgent(depth=depth)
    elif name == "human":
        return HumanAgent()
    raise ValueError(f"Unknown agent '{name}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--white",  choices=["random", "minimax", "human"], default="random")
    parser.add_argument("--black",  choices=["random", "minimax", "human"], default="random")
    parser.add_argument("--depth",  type=int, default=3)
    parser.add_argument("--games",  type=int, default=1)
    parser.add_argument("--seed",   type=int, default=None)
    parser.add_argument("--quiet",  action="store_true")
    parser.add_argument("--ui",     action="store_true", help="Render a live chess UI window")
    parser.add_argument("--manual", action="store_true", help="Set white agent to human input mode")
    args = parser.parse_args()
    white_name = "human" if args.manual else args.white

    asyncio.run(
        run_match(
            white=build_agent(white_name, args.depth, args.seed),
            black=build_agent(args.black, args.depth, args.seed),
            n_games=args.games,
            verbose=not args.quiet,
            ui_enabled=args.ui,
            )
        )