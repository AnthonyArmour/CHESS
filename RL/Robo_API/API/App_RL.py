from RoboA2C_Mcts import A3CAgent
from flask import Flask, request
import numpy as np
from io import BytesIO
import flask


def bytes_to_array(b: bytes) -> np.ndarray:
    np_bytes = BytesIO(b)
    return np.load(np_bytes, allow_pickle=True)


def array_to_bytes(x: np.ndarray) -> bytes:
    np_bytes = BytesIO()
    np.save(np_bytes, x, allow_pickle=True)
    return np_bytes.getvalue()


app = Flask(__name__)

global model
model = A3CAgent(version="AWS_1")


@app.route("/api/Get_Data_From_A2C_MCTS", methods=["POST"])
def Run_A2C_MCTS():
    r = request.json
    episodes = r["episodes"]
    mcts_iterations = r["mcts_iterations"]
    load = bool(r["load"])

    states, actions, rewards, wins, losses = model.Run_Games(episodes, mcts_iterations)
    SAR = array_to_bytes(np.array([states, actions, rewards, wins, losses]))
    SAR = flask.make_response(SAR)
    SAR.headers.set("Content-Type", "application/octet_stream")
    return SAR


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
