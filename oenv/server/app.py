from openenv.core.env_server import create_app
from oenv.server.chess_oenv import ChessOpenEnv
from oenv.model import ChessOpenEnvAction, ChessOpenEnvObservation

app = create_app(ChessOpenEnv, ChessOpenEnvAction, ChessOpenEnvObservation, "chess_openenv")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)