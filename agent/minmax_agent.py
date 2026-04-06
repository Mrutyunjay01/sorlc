from agent.base_agent import BaseAgent
from envs.chess_env import ChessEnv, ChessObservation, ChessAction

# Sentinel scores — same scale as material_balance (centipawns)
_WIN  =  100_000
_LOSS = -100_000


def _alpha_beta(
    obs: ChessObservation,
    depth: int,
    alpha: float,
    beta: float,
    maximising: bool,
) -> float:
    # if terminal state, evaluate
    if depth == 0:
        return obs.evaluation

    # Fresh sim_env per call — no shared mutable state across recursion levels
    sim_env = ChessEnv()

    if maximising:
        best = -float("inf")
        for move in obs.legal_moves:
            _ = sim_env.reset(fen=obs.fen)
            step_result = sim_env.step(ChessAction(move_uci=move))

            score = (
                step_result.reward
                if step_result.done
                else _alpha_beta(
                    step_result.observation,
                    depth - 1,
                    alpha,
                    beta,
                    False,
                )
            )

            best  = max(best, score)
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return best

    else:
        best = float("inf")
        for move in obs.legal_moves:
            _ = sim_env.reset(fen=obs.fen)
            step_result = sim_env.step(ChessAction(move_uci=move))

            score = (
                step_result.reward
                if step_result.done
                else _alpha_beta(
                    step_result.observation,
                    depth - 1,
                    alpha,
                    beta,
                    True,
                )
            )

            best = min(best, score)
            beta = min(beta, best)
            if beta <= alpha:
                break
        return best


class MinimaxAgent(BaseAgent):

    def __init__(self, depth: int = 3):
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self.depth = depth

    def select_action(self, obs: ChessObservation) -> ChessAction:
        maximising = (obs.turn == "white")
        best_move  = None
        best_score = -float("inf") if maximising else float("inf")

        sim_env = ChessEnv()
        _ = sim_env.reset(fen=obs.fen) # reset to state of the observation

        print(f"evaluating {len(obs.legal_moves)} moves for board state: {obs.fen}")
        for move in obs.legal_moves:
            step_result = sim_env.step(ChessAction(move_uci=move))

            score = (
                step_result.reward # return the reward/evaluation from the step if terminal step
                if step_result.done
                else _alpha_beta(
                    step_result.observation, # state/obs from environment after action taken
                    depth=self.depth - 1,
                    alpha=-float("inf"),
                    beta=float("inf"),
                    maximising=not maximising,
                ) # continue with next step
            )

            if (maximising and score > best_score) or \
               (not maximising and score < best_score):
                best_score = score
                best_move  = move

            # reset env to initial observation state
            _ = sim_env.reset(obs.fen)

            # print(f"best move {best_move} with score {best_score}")

        return ChessAction(move_uci=best_move)

    @property
    def name(self) -> str:
        return f"MinimaxAgent(depth={self.depth})"