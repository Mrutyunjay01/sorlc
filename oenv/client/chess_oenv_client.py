from typing import Dict, Any
from oenv.model import ChessOpenEnvAction, ChessOpenEnvObservation, ChessOpenEnvState
from openenv.core.env_client import EnvClient, StepResult

class ChessOpenEnvClient(EnvClient[ChessOpenEnvAction, ChessOpenEnvObservation, ChessOpenEnvState]):

    # overrides
    def _step_payload(self, action: ChessOpenEnvAction):
        """ convert action to json payload (weird, why? why not just pass on the dataclass?)"""
        return {
            "move_uci": action.move_uci # make sure it's in the same data contract as server action model
        }
    
    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ChessOpenEnvObservation]:
        """ convert step result into observation """

        obs_data = payload.get("observation")
        obs = ChessOpenEnvObservation(
            fen=obs_data.get("fen", ""),
            legal_moves=obs_data.get("legal_moves", []),
            evaluation=obs_data.get("evaluation", 0.0),
            turn=obs_data.get("turn", "white"),
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
            meta_info=obs_data.get("meta_info", {})
        )

        return StepResult(
            observation=obs,
            reward=obs.reward,
            done=obs.done
        )
    
    def _parse_state(self, payload: Dict[str, Any]) -> ChessOpenEnvState:
        return ChessOpenEnvState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0)
        )