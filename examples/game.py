import argparse
import time
from agent.base_agent import BaseAgent
from agent.random_agent import RandomAgent
from agent.minmax_agent import MinimaxAgent
from envs.chess_env import ChessEnv

def play_game(
    white: BaseAgent,
    black: BaseAgent,
    verbose: bool = True,
) -> dict:
    env = ChessEnv()
    obs = env.reset()
    moves = []
    start = time.time()

    if verbose:
        print(f"\n{'='*50}")
        print(f"  {white.name} (White)  vs  {black.name} (Black)")
        print(f"{'='*50}")

    while True:
        agent = white if obs.turn == "white" else black
        action = agent.select_action(obs)
        obs = env.step(action=action)
        moves.append(action.move_uci)

        if obs.done:
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

    return {
        "outcome": winner,
        "outcome_str": outcome_str,
        "move_count": len(moves),
        "moves": moves,
        "duration_s": duration,
    }


def run_match(
    white: BaseAgent,
    black: BaseAgent,
    n_games: int = 1,
    verbose: bool = True,
):
    tally = {"white": 0, "black": 0, "draw": 0}
    total_moves = 0
    total_time = 0.0
    per_game_verbose = verbose and n_games == 1

    for game_num in range(1, n_games + 1):
        if n_games > 1:
            print(f"  Game {game_num}/{n_games}...", end=" ", flush=True)

        result = play_game(white, black, verbose=per_game_verbose)
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
    raise ValueError(f"Unknown agent '{name}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--white",  choices=["random", "minimax"], default="random")
    parser.add_argument("--black",  choices=["random", "minimax"], default="random")
    parser.add_argument("--depth",  type=int, default=3)
    parser.add_argument("--games",  type=int, default=3)
    parser.add_argument("--seed",   type=int, default=None)
    parser.add_argument("--quiet",  action="store_true")
    args = parser.parse_args()

    run_match(
        white=build_agent(args.white, args.depth, args.seed),
        black=build_agent(args.black, args.depth, args.seed),
        n_games=args.games,
        verbose=not args.quiet,
    )