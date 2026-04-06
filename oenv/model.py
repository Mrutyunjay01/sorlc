from openenv.core.env_server import Action, Observation, State

class ChessOpenEnvAction(Action):
    move_uci: str

class ChessOpenEnvObservation(Observation):
    """ represents an observation (not a state) from the env """
    fen         : str
    legal_moves : list[str]
    evaluation  : int
    turn        : str
    meta_info   : dict
    
class ChessOpenEnvState(State):
    episode_id: str | None = None
    step_count: int = 0
    fen: str = ""
    turn: str = "white"