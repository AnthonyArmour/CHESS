import requests
import numpy as np
from io import BytesIO


# MOVE TO DEVELOPMENT FOLDER


def bytes_to_array(b: bytes) -> np.ndarray:
    np_bytes = BytesIO(b)
    return np.load(np_bytes, allow_pickle=True)


def Mine_Servers(episodes, mcts_iterations, fit_iterations):

    data = {
        "episodes": str(episodes),
        "mcts_iterations": str(mcts_iterations),
        "fit_iterations": str(fit_iterations),
        "ip1": "",
        "ip2": ""
    }
    url = "http://ip:5000/api/Multi_Process_Mining"
    
    response = requests.post(url, json=data)
    print(response.content)


def Get_Weights():

    url = "http://ip:5000/api/Get_Weights"

    response = requests.post(url, json={"blank": "blank"})
    weights = bytes_to_array(response.content)

    actor_weights = weights[0]
    critic_weights = weights[1]

    from A2C_Model_2 import ActorCritic
    model = ActorCritic()
    actor, critic, _ = model.get_ActorCritic_ResConvNet()

    actor.set_weights(actor_weights)
    critic.set_weights(critic_weights)

    actor.save_weights("../Latest_Weights/Actor")
    critic.save_weights("../Latest_Weights/Critic")


if __name__ == "__main__":
    Mine_Servers(1, 200, 2)