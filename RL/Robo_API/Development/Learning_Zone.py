import sys

sys.path.insert(0, "/home/vagrant/CHESS/RL/Robo_API/API")

from A2C_Model_2 import ActorCritic
import numpy as np
import requests
from io import BytesIO
import pickle
import os



class LearningZone():

    def __init__(self):
        self.Model = ActorCritic()
        self.Actor, self.Critic, _ = self.Model.get_ActorCritic_ResConvNet()

    @staticmethod
    def array_to_bytes(x: np.ndarray) -> bytes:
        np_bytes = BytesIO()
        np.save(np_bytes, x, allow_pickle=True)
        return np_bytes.getvalue()

    @staticmethod
    def bytes_to_array(b: bytes) -> np.ndarray:
        np_bytes = BytesIO(b)
        return np.load(np_bytes, allow_pickle=True)

    def load(self, filename):
        """Loads object from pickle file"""
        try:
            with open(filename, "rb") as fh:
                obj = pickle.load(fh)
            return obj
        except Exception:
            return None

    def pkl(self, obj, filename):
        """Saves pickled object to .pkl file"""
        if filename.endswith(".pkl") is False:
            filename += ".pkl"
        with open(filename, "wb") as fh:
            pickle.dump(obj, fh)

    def save_weights(self):
        self.Actor.save_weights("../Weights/Actor")
        self.Critic.save_weights("../Weights/Critic")

    def save_SAR(self, states, actions, rewards):
        if os.path.exists("../Memory/states.pkl"):
            Pstates = self.load("../Memory/states.pkl")
            Pactions = self.load("../Memory/actions.pkl")
            Prewards = self.load("../Memory/rewards.pkl")
            states = np.concatenate((Pstates, states), axis=0)
            actions = np.concatenate((Pactions, actions), axis=0)
            rewards = np.concatenate((Prewards, rewards), axis=0)
        self.pkl(states, "../Memory/states.pkl")
        self.pkl(actions, "../Memory/actions.pkl")
        self.pkl(rewards, "../Memory/rewards.pkl")
        print("------Saved Sates {}, Actions {}, and Rewards {}------\n".format(states.shape, actions.shape, rewards.shape))

    def Collect_Games(self, ip, port, episodes=1, mcts_iterations=150, load=1):
        params = {
            "episodes": str(episodes),
            "mcts_iterations": str(mcts_iterations),
            "load": str(load)
        }
        url = "http://{}:{}/api/Get_Data_From_A2C_MCTS".format(ip, port)

        response = requests.post(url, json=params)
        SAR = self.bytes_to_array(response.content)
        states, actions, rewards, wins, losses = SAR
        print("Total Wins/Losses/W&L: {}/{}/{}\n".format(wins, losses, wins+losses))
        self.save_SAR(states, actions, rewards)
        return wins, losses

    def Fit_Memory(self, batch_size):
        states = self.load("../Memory/states.pkl")
        actions = self.load("../Memory/actions.pkl")
        rewards = self.load("../Memory/rewards.pkl")

        rand = np.random.choice(np.arange(rewards.shape[0]), (batch_size,), replace=False)

        states, rewards = states[rand, ...], rewards[rand, ...]
        value = self.Critic.predict(states)[:, 0]

        advantages = rewards - value
        self.Actor.fit(states, actions[rand, ...], sample_weight=advantages, epochs=1, verbose=True)
        self.Critic.fit(states, rewards, epochs=1, verbose=True)
        self.save_weights()

    def Test(self):
        wins, losses = self.Collect_Games("18.217.42.216", "5000", episodes=1, mcts_iterations=10, load=0)
        self.Fit_Memory(200)


