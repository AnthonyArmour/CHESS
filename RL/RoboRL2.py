import chess
import chess.engine
import random
import numpy as np
import os
from matplotlib import pyplot as plt
# import sys
import pickle

from tensorflow.python.autograph.operators.py_builtins import next_
from ActorCritic2 import ActorCritic
# from ChessModelTools_v7_ResNet import Tools


class TrainRoboRL():
    def __init__(self, episodes=200, lr=0.00001, version=None, loadfile=False):
        self.episodes = episodes
        self.max_avg = None
        self.version = version
        self.noOutcome = 0
        self.batch = 0
        self.gamma = 0.9
        self.engine = chess.engine.SimpleEngine.popen_uci("../stockfish_14_linux_x64_avx2/stockfish_14_x64_avx2")
        self.path = os.getcwd()
        self.actionSpace = self.load("../data/classes.pkl")
        self.IactionSpace = self.load("../data/inverted_classes.pkl")


        self.ActorDir = 'RL_Models/Actors/'
        if not os.path.exists(self.ActorDir): os.makedirs(self.ActorDir)
        self.Actor_file = '{}_Actor_{}'.format(version, lr)
        self.paths = {"actor": os.path.join(self.ActorDir, self.Actor_file)}

        self.CriticDir = 'RL_Models/Critics/'
        if not os.path.exists(self.CriticDir): os.makedirs(self.CriticDir)
        self.Critic_file = '{}_Critic_{}'.format(version, lr)
        self.paths["critic"] = os.path.join(self.CriticDir, self.Critic_file)


        if loadfile is False:
            self.Model = ActorCritic(len(self.actionSpace), lr=lr)
        else:
            self.Model = ActorCritic(len(self.actionSpace), lr=lr, paths=self.paths)

        self.board = chess.Board()
        self.illegal_moves = 0
        self.quickTrain = False
        self.e = 0
        self.broken = 0
        self.cached_reward_state = 0
        self.state, self.action, self.reward = None, None, None
        self.scores, self.Episodes, self.average = [], [], []
        self.next_state, self.done = None, None

        self.info = {
            "exploit": 0,
            "explore": 0
        }
        # self.legal_cnt = {
        #     "illegal": 0,
        #     "legal": 0,
        #     "NetLegal": 0
        # }

    def save(self):
        self.Model.Actor.save(self.paths["actor"])
        self.Model.Critic.save(self.paths["critic"])
        self.PlotProg()

    def LogAvg(self, score, episode):
        self.scores.append(score)
        self.Episodes.append(episode)
        self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))
        return self.average[-1]

    def PlotProg(self):
        plt.plot(self.Episodes, self.scores, 'b')
        plt.plot(self.Episodes, self.average, 'r')
        plt.suptitle("Score Over Episodes")
        plt.ylabel('Score', fontsize=18)
        plt.xlabel('Steps', fontsize=18)
        if not os.path.exists("RL_Models/Graphs"): os.makedirs("RL_Models/Graphs")
        try:
            plt.savefig("RL_Models/Graphs/{}_ScoreOT.png".format(self.version))
        except:
            pass

    def load(self, filename):
        """Loads object from pickle file"""
        try:
            with open(filename, "rb") as fh:
                obj = pickle.load(fh)
            return obj
        except Exception:
            return None

    def fen_to_board(self, fen):
        pieces = {
            "p": 5, "P": -5, "b": 15, "B": -15, "n": 25, "N": -25,
            "r": 35, "R": -35, "q": 45, "Q": -45, "k": 55, "K": -55
        }
        # pieces = {
        #     "p": 5, "P": 105, "b": 15, "B": 115, "n": 25, "N": 125,
        #     "r": 35, "R": 135, "q": 45, "Q": 145, "k": 55, "K": 155
        # }
        blank, slash = 0, 0
        samples = np.ones((1, 64))
        # samples = np.ones((64, 1))
        for x, c in enumerate(fen):
            if c == " ":
                break
            if c.isdigit():
                blank += int(c) - 1
                continue
            if c == "/":
                slash += 1
                continue
            samples[0][x+blank-slash] = pieces[c] + 1
            # samples[x+blank-slash][0] = pieces[c]
        samples = (np.reshape(samples, (1, 8, 8, 1))).astype(np.float32)
        # print(samples.shape)
        return samples


    def takeAction(self):
        self.done = False
        Action = chess.Move.from_uci(self.IactionSpace[self.action])
        # print("\n\n\n")
        # a = self.IactionSpace[action]
        # for mv in list(self.board.legal_moves):
        #     if str(mv) == a:
        #         print("LEGAL MOVE")
        if Action in self.board.legal_moves:
            self.broken = 0
            # print("TRUE")

            self.board.push(Action)
            # print("\n", self.board)
            if self.board.is_checkmate():
                self.reward = 50
                self.done = True
                # self.next_state = self.state
            elif self.board.outcome() is not None:
                self.done = True
                self.noOutcome += 1
                self.reward = 0
            else:
                result = self.engine.play(self.board,chess.engine.Limit(time=0.000001))
                self.board.push(result.move)
                # print("\n", self.board)
                self.next_state = self.fen_to_board(self.board.fen())

                reward_state = np.sum(self.next_state) - 64
                self.reward = reward_state - self.cached_reward_state
                self.cached_reward_state = reward_state

                # reward = self.engine.PovScore()
                if self.board.is_checkmate():
                    self.reward = -50
                    self.done = True
                    # self.next_state = None
            # self.legal_cnt["legal"] += 1
            # self.legal_cnt["NetLegal"] += 1
        else:
            # print("here")
            self.broken += 1
            # self.reward = -1
            # self.next_state = self.state
            # self.illegal_moves += 1

            # self.legal_cnt["illegal"] += 1
            # if self.illegal_moves == 100:
            #     self.illegal_moves = 0
            #     done = True
            #     self.quickTrain = True
            # else:
            #     self.quickTrain = False
        # return next_state, reward, done

    def trainGame(self):
        # reshape memory to appropriate shape for training
        # states = np.vstack(self.Model.states)
        # states = self.Model.states
        # actions = np.vstack(self.Model.actions)

        action_onehot = np.zeros([1, self.Model.action_space])
        action_onehot[0, self.action] = 1.0

        # Compute discounted rewards
        # Critic learns to fit this
        # Real reward found here to be learned from
        # TD targets
        # discounted_r = self.Model.discount_rewards(self.Model.rewards)

        # Get Critic network predictions
        value = self.Model.Critic.predict(self.state)[0]
        value_next = self.Model.Critic.predict(self.next_state)[0]

        target = self.reward + self.gamma*value_next*(1-int(self.done))
        advantage = target - value

        # Compute advantages / TD_Error
        # how Actor decides how good or bad action is
        # prediction error of agent
        # advantages = discounted_r - values
        # training Actor and Critic networks
        self.Model.Actor.fit(self.state, action_onehot, sample_weight=advantage, epochs=1, verbose=0)
        self.Model.Critic.fit(self.state, target, epochs=1, verbose=0)
        # self.Model.states, self.Model.actions, self.Model.rewards = None, [], []
        # reset training memory

    def getAction(self):
        # rand = random.random()
        # if rand < self.e/2100 or rand < 0.6:
        #     prediction = self.Model.Actor.predict(state)[0]
        #     # return np.random.choice(self.Model.action_space, p=prediction)
        #     return np.argmax(prediction)
        # else:
        #     prediction = np.random.choice(list(self.board.legal_moves))
        #     self.legal_cnt["NetLegal"] -= 1
        #     return self.actionSpace[str(prediction)]
        prediction = self.Model.Actor.predict(self.state)[0]
        if random.random() < 0.96:
            self.info["exploit"] += 1
            legal_moves = np.zeros(prediction.shape[0])
            for mv in list(self.board.legal_moves):
                legal_moves[self.actionSpace[str(mv)]] = 1
            prediction = prediction * legal_moves
            self.action = np.argmax(prediction)
        else:
            self.info["explore"] += 1
            prediction = np.random.choice(list(self.board.legal_moves))
            self.action = self.actionSpace[str(prediction)]

    def reset(self):
        self.engine.quit()
        self.board = chess.Board()
        self.engine = chess.engine.SimpleEngine.popen_uci("../stockfish_14_linux_x64_avx2/stockfish_14_x64_avx2")

    def RunTrainingSession(self):
        # FixState = False

        while self.e <= self.episodes:

            # print("\nGame:", episode, end=" ")

            self.done, score, SAVE = False, 0, ""
            # board = chess.Board() #give whatever starting position here
            #push first move
            result = self.engine.play(self.board,chess.engine.Limit(time=0.01))
            self.board.push(result.move)
            # print("\n\n\n\n{}\n{}".format(self.e, self.board))
            self.state = self.fen_to_board(self.board.fen())

            moves = 0
            self.info["exploit"], self.info["explore"] = 0, 0
            self.cached_reward_state = 0
            # self.legal_cnt["illegal"], self.legal_cnt["legal"], self.legal_cnt["NetLegal"] = 0, 0, 0
            while not self.board.is_game_over() and self.done is False:

                # if moves % 1000 == 0:
                #     print("{} moves | {} illegal_moves | {} legal moves".format(moves, self.legal_cnt["illegal"], self.legal_cnt["legal"]))

                self.getAction()
                self.takeAction()

                if self.broken == 0:
                    self.trainGame()
                    self.state = self.next_state
                    score += self.reward
                    moves += 1
                else:
                    break



            if self.broken == 0:
                self.e += 1
                self.batch += moves
                average = self.LogAvg(score, self.e)
            if self.max_avg is None or score >= self.max_avg:
                self.max_avg = score
            #     # self.save()
            #     # SAVE = "SAVING"
            # else:
            #     SAVE = ""
            if self.e % 100 == 0:
                print("Game: {}/{}  |  {} Moves in last mod batches".format(self.e, self.episodes, self.batch))
                print("score: {}, average: {:.2f}, Best Score: {} {}".format(score, average, self.max_avg, SAVE))
                # print(self.Model.scores)
                print("{} moves | Exploit: {} | Explore: {} | {} NoOutcome moves\n\n".format(moves, float(self.info["exploit"]/moves), float(self.info["explore"]/moves), self.noOutcome))
                self.batch, self.noOutcome = 0, 0


            self.reset()

        self.save()
        self.engine.quit()
