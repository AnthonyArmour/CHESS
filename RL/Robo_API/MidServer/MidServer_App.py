from tensorflow.python.keras.layers.convolutional import Cropping1D
from Learning_Zone import LearningZone
from flask import Flask, request, Response
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

global LZ
LZ = LearningZone()


@app.route("/api/save_initial_weights", methods=["POST"])
def initial_save():
    r = request.json
    LZ.Actor.save_weights("../Weights/Actor")
    LZ.Actor.save_weights("../Weights/Coach")
    LZ.Critic.save_weights("../Weights/Critic")
    return Response(response="Saved!", status=200, mimetype="application/octet_stream")


@app.route("/api/Get_Weights", methods=["POST"])
def Get_Weights():
    r = request.json
    LZ.Actor.load_weights("../Weights/Actor")
    actor_weights = LZ.Actor.get_weights()
    LZ.Critic.load_weights("../Weights/Critic")
    critic_weights = LZ.Critic.get_weights()

    weights = np.array([actor_weights, critic_weights], dtype="float32")
    weights = flask.make_response(array_to_bytes(weights))
    weights.headers.set("Content-Type", "application/octet_stream")
    return weights


@app.route("/api/Multi_Process_Mining", methods=["POST"])
def Mine():
    r = request.json
    episodes = int(r["episodes"])
    mcts_iterations = int(r["mcts_iterations"])
    fit_iterations = int(r["fit_iterations"])
    ip1 = r["ip1"]
    ip2 = r["ip2"]

    for x in range(fit_iterations):
        LZ.Actor.load_weights("../Weights/Coach")
        coach_weights = LZ.Actor.get_weights()
        LZ.Actor.load_weights("../Weights/Actor")
        actor_weights = LZ.Actor.get_weights()
        critic_weights = LZ.Critic.get_weights()

        weights = np.array([actor_weights, critic_weights, coach_weights], dtype='float32')

        wins, losses, samples = LZ.MultiProcess_Mining(episodes, mcts_iterations, weights, ip1, ip2)

        print("Wins: {} Losses: {} Samples: {}".format(wins, losses, samples))

        if ((wins/(wins+losses)) > 0.6):
            LZ.Actor.save_weights("../Weights/Coach")

        batch_size = int(samples*.95)
        LZ.Fit_Memory(batch_size)

    return Response(response="Done!", status=200, mimetype="application/octet_stream")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
