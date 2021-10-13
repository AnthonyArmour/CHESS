import enum
import chess
import chess.engine
import random
import numpy as np
import os
from matplotlib import pyplot as plt
import pickle
from tensorflow.python.autograph.operators.py_builtins import next_
from ActorCritic7 import ActorCritic


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
        self.actionSpaceL = len(self.actionSpace)
        self.IactionSpace = self.load("../data/inverted_classes.pkl")


        self.ActorDir = 'RL_Models/Actors/'
        if not os.path.exists(self.ActorDir): os.makedirs(self.ActorDir)
        self.Actor_file = '{}_Actor_{}'.format(version, lr*10)
        self.paths = {"actor": os.path.join(self.ActorDir, self.Actor_file)}

        self.CriticDir = 'RL_Models/Critics/'
        if not os.path.exists(self.CriticDir): os.makedirs(self.CriticDir)
        self.Critic_file = '{}_Critic_{}'.format(version, lr*10)
        self.paths["critic"] = os.path.join(self.CriticDir, self.Critic_file)

        self.PolicyDir = 'RL_Models/Policies/'
        if not os.path.exists(self.PolicyDir): os.makedirs(self.PolicyDir)
        self.Policy_file = '{}_Policy_{}'.format(version, lr*10)
        self.paths["policy"] = os.path.join(self.PolicyDir, self.Policy_file)

        self.CoachDir = 'RL_Models/Coaches/'
        if not os.path.exists(self.CoachDir): os.makedirs(self.CoachDir)
        self.Coach_file = '{}_Coach_{}'.format(version, lr*10)
        self.paths["coach"] = os.path.join(self.CoachDir, self.Coach_file)


        Model = ActorCritic(self.actionSpaceL, lr=lr)
        if loadfile is False:
            self.Actor, self.Critic, self.Policy, self.Coach = Model.get_ActorCritic_ResConvNet()
        else:
            self.Actor, self.Critic, self.Policy, self.Coach = Model.get_ActorCritic_ResConvNet(paths=self.paths)
        del Model

        self.board = None
        self.illegal_moves = 0
        # self.quickTrain = False
        self.e = 0
        self.broken = 0
        self.legal_moves = None
        self.pred = None
        self.cached_reward_state = 0
        self.state, self.action, self.reward = None, None, None
        self.scores, self.Episodes, self.average = [], [], []
        self.next_state, self.done = None, False

        self.info = {
            "exploit": 0,
            "explore": 0
        }

    def save(self):
        self.Actor.save_weights(self.paths["actor"])
        self.Critic.save_weights(self.paths["critic"])
        self.Policy.save_weights(self.paths["policy"])
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
        blank, slash = 0, 0
        samples = np.ones((1, 64))
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
        # print((np.reshape(samples, (1, 8, 8))).astype(np.float32))
        samples = (np.reshape(samples, (1, 8, 8, 1))).astype(np.float32)
        # print(samples, "\n")
        return samples

    def flip_perspective(self, board):
        return -np.rot90(board, 2)

    def reset(self):
        # if self.engine:
        #     self.engine.quit()
        del self.board
        self.board = chess.Board()
        # self.engine = chess.engine.SimpleEngine.popen_uci("../stockfish_14_linux_x64_avx2/stockfish_14_x64_avx2")
        self.info["exploit"], self.info["explore"], self.done = 0, 0, False
        self.cached_reward_state, self.moves, self.score = 0, 0, 0

        result = self.engine.play(self.board,chess.engine.Limit(time=0.01))
        self.board.push(result.move)
        # print("\n\n\n\n{}\n{}".format(self.e, self.board))
        self.state = self.fen_to_board(self.board.fen())
        return ""

    def takeAction(self, Action):
        self.done = False
        # Action = chess.Move.from_uci(self.IactionSpace[self.action])
        if Action in self.board.legal_moves:
            self.broken = 0
            # print("TRUE")

            self.board.push(Action)
            # print("\n", self.board)
            if self.board.is_checkmate():
                print("\nWon Game?")
                print(self.board)
                print(self.board.outcome().winner, "\n")
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

                reward_state = (np.sum(self.next_state) - 64) * 0.1
                self.reward = reward_state - self.cached_reward_state
                self.cached_reward_state = reward_state

                # reward = self.engine.PovScore()
                if self.board.is_checkmate():
                    self.reward = -50
                    self.done = True
                    # self.next_state = None
        else:
            self.broken += 1
            print("brk")
            print("Bug move: {}: {}".format(Action, self.pred[self.actionSpace[str(Action)]]))
            for mv in self.legal_moves:
                print("{}: {} | ".format(mv, self.pred[self.actionSpace[mv]]), end="")
            print("\nOrdered predictions - ", end="")
            for m in self.a:
                print("{}: {} | ".format(self.IactionSpace[m], self.pred[m]))


    def trainGame(self):

        action_onehot = np.zeros([1, self.actionSpaceL])
        action_onehot[0, self.action] = 1.0


        value = self.Critic.predict(self.state)[0]
        value_next = self.Critic.predict(self.next_state)[0]

        target = self.reward + self.gamma*value_next*(1-int(self.done))
        advantage = target - value

        self.Actor.fit([self.state, advantage], action_onehot, epochs=1, verbose=0) #, sample_weight=advantage
        self.Critic.fit(self.state, target, epochs=1, verbose=0)


    def getAction(self, coach=False):
        # errn = np.seterr(all='raise')
        if coach is True:
            policy = self.Coach
        else:
            policy = self.Policy

        bugValue = False
        self.legal_moves = [str(mv) for mv in list(self.board.legal_moves)]

        if random.random() < (0.95 + ((self.e/self.episodes)*0.03)):
            prediction = policy.predict(self.state)[0]
            legal_moves = np.zeros(prediction.shape[0])

            for mv in self.legal_moves:
                if self.actionSpace[mv] == 0:
                    bugValue = True
                legal_moves[self.actionSpace[mv]] = 1
            prediction = prediction.astype(np.float128) + legal_moves.astype(np.float128)
            # for p in prediction:
            #     print(p)

            self.pred = prediction
            a = np.argsort(-prediction, axis=0)[:5]
            self.a = a

            for i in a:
                if self.IactionSpace[i] in self.legal_moves:
                    self.action = i
                    break

            Action = chess.Move.from_uci(self.IactionSpace[self.action])
            self.info["exploit"] += 1
            return Action
        elif coach is False:
            self.info["explore"] += 1
            prediction = np.random.choice(self.legal_moves)
            self.action = self.actionSpace[prediction]
            return chess.Move.from_uci(self.IactionSpace[self.action])


    def RunTrainingSession(self):
        broke, stail = 0, 0
        # print("\n\n\n\n", self.actionSpace["f2c2"], "\n\n\n\n")

        while self.e <= self.episodes:

            if broke > 100:
                print(self.board)
                print("BROKEN. stail: {}".format(stail))
                self.engine.quit()
                exit(1)

            SAVE = self.reset()

            while self.done is False:

                Action = self.getAction()
                self.takeAction(Action)

                if self.broken != 0:
                    break
                else:
                    self.trainGame()
                    self.state = self.next_state
                    self.score += self.reward
                    self.moves += 1



            if self.broken == 0:
                self.e += 1
                broke = 0
                stail = 0
                self.batch += self.moves
                average = self.LogAvg(self.score, self.e)
            else:
                broke += 1
                continue
            if self.max_avg is None or self.score >= self.max_avg:
                self.max_avg = self.score
                if self.e > 200:
                    print("Max score:\n", self.board)

            if self.e % 20 == 0:
                print("Game: {}/{}  |  {} Moves in last mod batches".format(self.e, self.episodes, self.batch))
                print("score: {}, average: {:.2f}, Best Score: {} {}".format(self.score, average, self.max_avg, SAVE))
                # print(self.Model.scores)
                print("{} moves | Exploit: {} | Explore: {} | {} NoOutcome moves\n\n".format(self.moves, float(self.info["exploit"]/self.moves), float(self.info["explore"]/self.moves), self.noOutcome))
                self.batch, self.noOutcome = 0, 0
                print("broken: {} | stail: {}".format(broke, stail))
            # if self.e % 1500 == 0:
            #     self.save()



        self.save()
        self.engine.quit()


