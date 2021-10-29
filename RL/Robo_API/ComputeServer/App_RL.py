from RoboA2C_Mcts import A3CAgent
from multiprocessing import Process, Pipe
from flask import Flask, request
import numpy as np
from io import BytesIO
import flask
import tensorflow as tf
import sys


def bytes_to_array(b: bytes) -> np.ndarray:
    np_bytes = BytesIO(b)
    return np.load(np_bytes, allow_pickle=True)


def array_to_bytes(x: np.ndarray) -> bytes:
    np_bytes = BytesIO()
    np.save(np_bytes, x, allow_pickle=True)
    return np_bytes.getvalue()


def MultiProcess_Mining(model, episodes, mcts_iterations, weights):
    parent_connections, processes = [], []
    States, Actions, Rewards = [], [], []
    Wins, Losses = 0, 0

    # lock = Lock()

    for x in range(3):
        parent, child = Pipe()
        parent_connections.append(parent)

        process = Process(
            target=model.Run_Games, args=(child, episodes, mcts_iterations, weights)
            )
        processes.append(process)
    

    for process in processes:
        process.start()
    
    for process in processes:
        process.join()

    for parent_conn in parent_connections:
        states, actions, rewards, wins, losses = parent_conn.recv()
        States.append(states)
        Actions.append(actions)
        Rewards.append(rewards)
        Wins += wins
        Losses += losses
    
    States = np.concatenate(States, axis=0)
    Actions = np.concatenate(Actions, axis=0)
    Rewards = np.concatenate(Rewards, axis=0)

    
    return States, Actions, Rewards, Wins, Losses


app = Flask(__name__)

global model
global graph
graph = tf.compat.v1.get_default_graph()
model = A3CAgent(graph, version="AWS_1")


@app.route("/api/Get_Data_From_A2C_MCTS", methods=["POST"])
def Run_A2C_MCTS():
    r = bytes_to_array(request.data)
    episodes = r[0]
    mcts_iterations = r[1]
    weights = r[2]

    states, actions, rewards, wins, losses = MultiProcess_Mining(model, episodes, mcts_iterations, weights)
    SAR = array_to_bytes(np.array([states, actions, rewards, wins, losses]))
    SAR = flask.make_response(SAR)
    SAR.headers.set("Content-Type", "application/octet_stream")
    return SAR


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(sys.argv[1]), debug=True)
