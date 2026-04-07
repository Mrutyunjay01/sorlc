from agent.base_agent import BaseAgent
from envs.chess_env import ChessEnv, ChessObservation, ChessAction


def _alpha_beta(
    obs: ChessObservation,
    depth: int,
    alpha: float,
    beta: float,
    maximising: bool,
) -> float:
    # Cumulative reward search from current node onward.
    if depth == 0 or obs.done or not obs.legal_moves:
        return obs.evaluation # return evaluation

    # Fresh sim_env per call — no shared mutable state across recursion levels
    sim_env = ChessEnv()

    if maximising:
        best = -float("inf")
        for move in obs.legal_moves:
            _ = sim_env.reset(fen=obs.fen)
            step_result = sim_env.step(ChessAction(move_uci=move))

            score = step_result.reward
            if not step_result.done:
                score += _alpha_beta(
                    step_result.observation,
                    depth - 1,
                    alpha,
                    beta,
                    False,
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

            score = step_result.reward
            if not step_result.done:
                score += _alpha_beta(
                    step_result.observation,
                    depth - 1,
                    alpha,
                    beta,
                    True,
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
        if obs.done:
            raise ValueError("cannot select action from terminal observation")
        if not obs.legal_moves:
            raise ValueError("cannot select action when no legal moves are available")

        maximising = (obs.turn == "white")
        best_move  = None
        best_score = -float("inf") if maximising else float("inf")

        sim_env = ChessEnv()
        _ = sim_env.reset(fen=obs.fen) # reset to state of the observation

        print(f"evaluating {len(obs.legal_moves)} moves for board state: {obs.fen}")
        for move in obs.legal_moves:
            step_result = sim_env.step(ChessAction(move_uci=move))

            score = step_result.reward
            if not step_result.done:
                score += _alpha_beta(
                    step_result.observation,
                    depth=self.depth - 1,
                    alpha=-float("inf"),
                    beta=float("inf"),
                    maximising=not maximising,
                )

            if (maximising and score > best_score) or \
               (not maximising and score < best_score):
                best_score = score
                best_move  = move

            # reset env to initial observation state
            _ = sim_env.reset(obs.fen)

            # print(f"best move {best_move} with score {best_score}")

        if best_move is None:
            raise RuntimeError("failed to choose a move from legal move list")
        return ChessAction(move_uci=best_move)

    @property
    def name(self) -> str:
        return f"MinimaxAgent(depth={self.depth})"