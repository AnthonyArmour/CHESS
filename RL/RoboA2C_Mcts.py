import os
import pickle
import random
import zlib
import requests
import io
import numpy as np
import chess
from keras import backend as K
import tensorflow as tf
from io import BytesIO
import matplotlib.pyplot as plt
from tensorflow.python.ops.gen_dataset_ops import cache_dataset
from A2C_Model_2 import ActorCritic
from MCTS_DRL import MCTS


class A3CAgent():

    def __init__(self, episodes=200, lr=0.00001, version=None, loadfile=False, oponent_benchmark=None):
        self.episodes = episodes
        self.gamma = 0.9
        self.end_game_reward = 0
        self.WHITE = False
        self.BLACK = True
        self.version = version
        self.max_avg = self.load("RL_Models/Graphs/{}_Max_Avg.pkl".format(self.version))
        # self.lock = Lock()

        self.wins, self.losses = 0.000001, 0.000001
        self.plot = 0
        self.API_MEM = "/api/memory_buffer"
        self.API_STATES = "/api/random_states_query"
        self.API_ACTIONS = "/api/random_actions_query"
        self.API_REWARDS = "/api/random_rewards_query"
        self.API_SAVE = "/api/save"
        self.API_LOAD = "/api/load"
        self.url = "http://192.168.0.37:5000"
        # self.engine = chess.engine.SimpleEngine.popen_uci("../stockfish_14_linux_x64_avx2/stockfish_14_x64_avx2")
        self.path = os.getcwd()
        self.actionSpace = self.load("../data/classes.pkl")
        self.actionSpaceL = len(self.actionSpace)
        mirror = self.load("../data/mirror_classes.pkl")
        IactionSpace = self.load("../data/inverted_classes.pkl")
        Imirror = {}
        for k, v in mirror.items():
            Imirror[v] = k

        self.ActorDir = 'RL_Models/Actors/'
        if not os.path.exists(self.ActorDir): os.makedirs(self.ActorDir)
        self.Actor_file = '{}_Actor_{}'.format(version, lr*10)
        self.paths = {"actor": os.path.join(self.ActorDir, self.Actor_file)}

        self.CriticDir = 'RL_Models/Critics/'
        if not os.path.exists(self.CriticDir): os.makedirs(self.CriticDir)
        self.Critic_file = '{}_Critic_{}'.format(version, lr*10)
        self.paths["critic"] = os.path.join(self.CriticDir, self.Critic_file)

        self.CoachDir = 'RL_Models/Coaches/'
        if not os.path.exists(self.CoachDir): os.makedirs(self.CoachDir)
        self.Coach_file = '{}_Coach_{}'.format(version, lr*10)
        self.paths["coach"] = os.path.join(self.CoachDir, self.Coach_file)


        Model = ActorCritic(self.actionSpaceL, lr=lr)
        if loadfile is False:
            Actor, Critic, Coach = Model.get_ActorCritic_ResConvNet()
        else:
            Actor, Critic, Coach = Model.get_ActorCritic_ResConvNet(paths=self.paths)
            self.wins = self.load("RL_Models/Graphs/{}_Wins.pkl".format(self.version))
            self.losses = self.load("RL_Models/Graphs/{}_Losses.pkl".format(self.version))
        
        del Model

        self.MCTS = MCTS(
            IactionSpace, self.actionSpace, Imirror, mirror, recursion_limit=10,
            iterations=150, Actor=Actor, Value_Net=Critic, Oponent=Coach
            )
        del Imirror

        if oponent_benchmark is not None:
            self.MCTS.Oponent.save_weights(self.CoachDir+oponent_benchmark)

        self.illegal_moves = 0
        self.broken = 0
        self.scores, self.Episodes, self.average = [], [], []
        # self.engine = chess.engine.SimpleEngine.popen_uci("../stockfish_14_linux_x64_avx2/stockfish_14_x64_avx2")


    @staticmethod
    def array_to_bytes(x: np.ndarray) -> bytes:
        np_bytes = BytesIO()
        np.save(np_bytes, x, allow_pickle=True)
        return np_bytes.getvalue()

    @staticmethod
    def bytes_to_array(b: bytes) -> np.ndarray:
        np_bytes = BytesIO(b)
        return np.load(np_bytes, allow_pickle=True)

    def Update_API_MemBuffer(self, actions, states, rewards):
        Data = {}

        Data["states"] = states.tolist()
        Data["rewards"] = rewards.tolist()
        Data["actions"] = actions.tolist()
        response = requests.post(self.url+self.API_MEM, json=Data, verify=False)
        return response.content

    def Query_Random_ASR(self, batch_size=20):
        states = requests.post(self.url+self.API_STATES, json={"batch_size": str(batch_size)}, verify=False)
        actions = requests.post(self.url+self.API_ACTIONS, json={"batch_size": str(batch_size)}, verify=False)
        rewards = requests.post(self.url+self.API_REWARDS, json={"batch_size": str(batch_size)}, verify=False)
        return self.bytes_to_array(states.content), self.bytes_to_array(actions.content), self.bytes_to_array(rewards.content)

    def Save_API_Remote_Mem(self):
        saved = requests.post(self.url+self.API_SAVE, verify=False)

    def Load_API_Remote_Mem(self):
        loaded = requests.post(self.url+self.API_LOAD, verify=False)

    def LogAvg(self, score, episode):
        self.scores.append(score)
        self.Episodes.append(episode)
        self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))
        return self.average[-1]

    def discount_rewards(self, reward):
        # Compute the gamma-discounted rewards over an episode
        gamma = 0.99    # discount rate
        ep = 0.00000001
        running_add = 0
        discounted_r = np.zeros_like(reward)
        for i in reversed(range(0,len(reward))):
            running_add = running_add * gamma + reward[i]
            discounted_r[i] = running_add

        discounted_r = discounted_r - np.mean(discounted_r) # normalizing the result
        discounted_r = (discounted_r+ep) / (np.std(discounted_r)+ep) # divide by standard deviation
        return discounted_r

    def TrainingReplay(self, states, actions, rewards, verbose=0):

        # action_onehot = np.zeros([1, self.actionSpaceL])
        # action_onehot[0, self.action] = 1.0

        # states = np.vstack(states)
        # actions = np.vstack(actions)
        
        # discounted_r = self.discount_rewards(rewards)
        discounted_r = rewards

        value = self.MCTS.Value_Net.predict(states)[:, 0]
        # value_next = self.Critic.predict(self.next_state)[0]

        advantages = discounted_r - value

        self.MCTS.Actor.fit(states, actions, sample_weight=advantages, epochs=1, verbose=verbose) #, sample_weight=advantage
        self.MCTS.Value_Net.fit(states, discounted_r, epochs=1, verbose=verbose)
 

    def save(self):
        self.MCTS.Actor.save_weights(self.paths["actor"])
        self.MCTS.Value_Net.save_weights(self.paths["critic"])
        self.MCTS.Oponent.save_weights(self.paths["coach"])
        self.pkl(self.wins, "RL_Models/Graphs/{}_Wins.pkl".format(self.version))
        self.pkl(self.losses, "RL_Models/Graphs/{}_Losses.pkl".format(self.version))
        self.PlotProg()

    def load(self, filename):
        """Loads object from pickle file"""
        try:
            with open(filename, "rb") as fh:
                obj = pickle.load(fh)
            return obj
        except Exception:
            return None

    def PlotProg(self):
        if os.path.exists("RL_Models/Graphs/{}_Scores.pkl".format(self.version)):
            self.scores = self.load("RL_Models/Graphs/{}_Scores.pkl".format(self.version)) + self.scores
            self.Episodes = self.load("RL_Models/Graphs/{}_Episodes.pkl".format(self.version)) + self.Episodes
            self.average = self.load("RL_Models/Graphs/{}_Average.pkl".format(self.version)) + self.average
        plt.plot(self.Episodes, self.scores, 'b')
        plt.plot(self.Episodes, self.average, 'r')
        plt.suptitle("Score Over Episodes")
        plt.ylabel('Score', fontsize=18)
        plt.xlabel('Steps', fontsize=18)
        if not os.path.exists("RL_Models/Graphs"): os.makedirs("RL_Models/Graphs")
        try:
            plt.savefig("RL_Models/Graphs/{}_{}-{}_ScoreOT_{}.png".format(self.version, self.wins, self.losses, self.plot))
            self.pkl(self.scores, "RL_Models/Graphs/{}_Scores.pkl".format(self.version))
            self.pkl(self.Episodes, "RL_Models/Graphs/{}_Episodes.pkl".format(self.version))
            self.pkl(self.average, "RL_Models/Graphs/{}_Average.pkl".format(self.version))
            self.Episodes, self.scores, self.average = [], [], []
            self.plot += 1
        except:
            pass

    def pkl(self, obj, filename):
        """Saves pickled object to .pkl file"""
        if filename.endswith(".pkl") is False:
            filename += ".pkl"
        with open(filename, "wb") as fh:
            pickle.dump(obj, fh)

    def reset(self):
        board = chess.Board()
        self.MCTS.board = chess.Board()
        self.MCTS.Node, self.MCTS.temp = None, 1
        score = 0
        mv = random.choice(list(board.legal_moves))
        board.push(mv)
        self.MCTS.feed(mv)
        state = self.MCTS.fen_to_board(board.fen())
        return "", board, score, state, False

    def takeAction(self, Action, board, idx):
        done = False
        # print("CHOSEN ACTION:\n", str(Action))
        # for mv in board.legal_moves:
        #     print("Legal Moves\n", str(mv))

        if Action in board.legal_moves:
            self.broken = 0
            board.push(Action)
            self.MCTS.feed(Action, idx)

            if board.is_checkmate():
                self.wins += 1
                self.end_game_reward = 1
                done = True
                next_state = None

            elif board.outcome() is not None:
                done = True
                next_state = None
                self.end_game_reward = (np.sum(self.MCTS.fen_to_board(board.fen()))-64)*0.001

            else:
                result, idx = self.MCTS.act(turn=self.WHITE)
                board.push(result)
                self.MCTS.feed(result, idx)
                next_state = self.MCTS.fen_to_board(board.fen())

                if board.is_checkmate():
                    self.losses += 1
                    self.end_game_reward = -1
                    # print("Loss Score - ", (np.sum(self.MCTS.fen_to_board(board.fen()))-64)*0.001)
                    done = True
                    next_state = None
                elif board.outcome() is not None:
                    done = True
                    self.end_game_reward = (np.sum(self.MCTS.fen_to_board(board.fen()))-64)*0.001
                    # print("End game result reward", self.end_game_reward)
                    next_state = None
            return next_state, board, done
        else:
            self.broken += 1
            print("brk")
            return None, None, True
        

    def TrainingSession(self, test=False, oponent_benchmark=None):
        broke = 0
        e = 0
        if test:
            self.Coach.load_weights(self.CoachDir+oponent_benchmark)
            self.Actor.load_weights(self.ActorDir+"CurrentBest")

        while e <= self.episodes:

            if broke > 100:
                print("BROKEN")
                exit(1)

            SAVE, board, score, state, done = self.reset()
            states, actions, mv_cnt = [], [], 0

            while done is False:

                # Action, actionIdx = self.getAction(board, state, e)
                Action, idx = self.MCTS.act(turn=self.BLACK)
                next_state, board, done = self.takeAction(Action, board, idx)

                if self.broken != 0:
                    break
                else:
                    states.append(state)
                    action_onehot = np.zeros([self.actionSpaceL])
                    action_onehot[self.actionSpace[str(Action)]] = 1
                    actions.append(action_onehot)

                    mv_cnt += 1
                    state = next_state
                
                if mv_cnt > 12 and self.MCTS.temp >= 0.01:
                    if self.MCTS.temp > 0.5:
                        self.MCTS.temp = 0.5
                        self.MCTS.iterations = 55
                    self.MCTS.temp -= 0.02

            score += self.end_game_reward
            state_memory = np.vstack(states)
            action_memory = np.vstack(actions)
            reward_memory = np.full((action_memory.shape[0],), self.end_game_reward)
            status = self.Update_API_MemBuffer(action_memory, state_memory, reward_memory)

            e += 1
            if not test:

                if self.broken == 0:
                    e += 1
                    average = self.LogAvg(score, e)
                else:
                    broke += 1
                    continue

                if self.max_avg is None or average >= self.max_avg:
                    self.max_avg = average
                    self.MCTS.Actor.save_weights(self.ActorDir+"CurrentBest")
                    self.pkl(self.max_avg, "RL_Models/Graphs/{}_Max_Avg.pkl".format(self.version))

                if e % 50 == 0:
                    self.Save_API_Remote_Mem()
                    states, actions, rewards = self.Query_Random_ASR(batch_size=800)
                    self.TrainingReplay(states, actions, rewards, verbose=True)
                    print("\nGame: {}/{}  ".format(e, self.episodes))
                    print("score: {}, average: {:.2f}, Best Score: {} {} | Total Wins/Losses/W&L: {}/{}/{}".format(score, average, self.max_avg, SAVE, self.wins, self.losses, self.wins+self.losses))

                if e % 500 == 0:
                    self.save()

                if ((self.wins/(self.wins+self.losses) > 0.55) and self.wins+self.losses > 150):
                    self.MCTS.Actor.save_weights(self.paths["actor"])
                    self.MCTS.Oponent.load_weights(self.paths["actor"])
                    self.PlotProg()
                    self.wins, self.losses, self.max_avg = 0.000001, 0.000001, 0
                    print("\n----Loading superior weights to Coach----\n")
            # elif e % 20 == 0 and e > 0:
            #     print(".....Querying API.....")
            #     T_states, T_actions, T_rewards = self.Query_Random_ASR(batch_size=4)
            #     print("Query states shape:", T_states.shape)
            #     print("Query aactions shape:", T_actions.shape)
            #     print("Query rewards shape", T_rewards.shape)


        if test:
            print("Total Wins/Losses/W&L: {}/{}/{}".format(self.wins, self.losses, self.wins+self.losses))
        else:
            self.save()


