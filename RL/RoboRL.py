import chess
import chess.engine
import random
import numpy as np
import os
# import sys
import pickle
from ActorCritic import ActorCritic
# from ChessModelTools_v7_ResNet import Tools


class TrainRoboRL():
    def __init__(self, episodes=200, saveFile=None, loadFile=None):
        self.episodes = episodes
        self.saveFile = saveFile
        self.loadFile = loadFile
        self.max_avg = None
        self.batch = 0
        self.engine = chess.engine.SimpleEngine.popen_uci("../stockfish_14_linux_x64_avx2/stockfish_14_x64_avx2")
        self.path = os.getcwd()
        self.actionSpace = self.load("../data/classes.pkl")
        self.IactionSpace = self.load("../data/inverted_classes.pkl")
        self.Model = ActorCritic(len(self.actionSpace))
        self.board = chess.Board()
        self.illegal_moves = 0
        self.quickTrain = False
        self.e = 0
        self.broken = 0
        self.cached_reward_state = 0
        self.info = {
            "exploit": 0,
            "explore": 0
        }
        # self.legal_cnt = {
        #     "illegal": 0,
        #     "legal": 0,
        #     "NetLegal": 0
        # }

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


    def takeAction(self, action, state):
        done = False
        Action = chess.Move.from_uci(self.IactionSpace[action])
        # print("\n\n\n")
        # a = self.IactionSpace[action]
        # for mv in list(self.board.legal_moves):
        #     if str(mv) == a:
        #         print("LEGAL MOVE")
        if Action in self.board.legal_moves:
            self.broken = 0
            # print("TRUE")

            self.board.push(Action)
            if self.board.is_checkmate():
                reward = 50
                done = True
                next_state = None
            elif self.board.outcome() is not None:
                done = "stail"
            else:
                result = self.engine.play(self.board,chess.engine.Limit(time=0.000001))
                self.board.push(result.move)
                next_state = self.fen_to_board(self.board.fen())

                reward_state = np.sum(next_state) - 64
                reward = reward_state - self.cached_reward_state
                self.cached_reward_state = reward_state

                # reward = self.engine.PovScore()
                if self.board.is_checkmate():
                    reward -= 50
                    done = True
                    next_state = None
            # self.legal_cnt["legal"] += 1
            # self.legal_cnt["NetLegal"] += 1
        else:
            # print("here")
            self.broken += 1
            reward = -1
            next_state = state
            self.illegal_moves += 1
            # self.legal_cnt["illegal"] += 1
            if self.illegal_moves == 100:
                self.illegal_moves = 0
                done = True
                self.quickTrain = True
            else:
                self.quickTrain = False
        return next_state, reward, done

    def trainGame(self):
        # reshape memory to appropriate shape for training
        # states = np.vstack(self.Model.states)
        states = self.Model.states
        actions = np.vstack(self.Model.actions)

        # Compute discounted rewards
        # Critic learns to fit this
        # Real reward found here to be learned from
        # TD targets
        discounted_r = self.Model.discount_rewards(self.Model.rewards)

        # Get Critic network predictions
        values = self.Model.Critic.predict(states)[:, 0]
        # Compute advantages / TD_Error
        # how Actor decides how good or bad action is
        # prediction error of agent
        advantages = discounted_r - values
        # training Actor and Critic networks
        self.Model.Actor.fit(states, actions, sample_weight=advantages, epochs=1, verbose=0)
        self.Model.Critic.fit(states, discounted_r, epochs=1, verbose=0)
        self.Model.states, self.Model.actions, self.Model.rewards = None, [], []
        # reset training memory

    def getAction(self, state):
        # rand = random.random()
        # if rand < self.e/2100 or rand < 0.6:
        #     prediction = self.Model.Actor.predict(state)[0]
        #     # return np.random.choice(self.Model.action_space, p=prediction)
        #     return np.argmax(prediction)
        # else:
        #     prediction = np.random.choice(list(self.board.legal_moves))
        #     self.legal_cnt["NetLegal"] -= 1
        #     return self.actionSpace[str(prediction)]
        prediction = self.Model.Actor.predict(state)[0]
        if random.random() < 0.98:
            self.info["exploit"] += 1
            legal_moves = np.zeros(prediction.shape[0])
            for mv in list(self.board.legal_moves):
                legal_moves[self.actionSpace[str(mv)]] = 1
            prediction = prediction * legal_moves
            return np.argmax(prediction)
        else:
            self.info["explore"] += 1
            prediction = np.random.choice(list(self.board.legal_moves))
            return self.actionSpace[str(prediction)]

    def reset(self):
        self.engine.quit()
        self.board = chess.Board()
        self.engine = chess.engine.SimpleEngine.popen_uci("../stockfish_14_linux_x64_avx2/stockfish_14_x64_avx2")

    def RunTrainingSession(self):
        # FixState = False

        for episode in range(self.episodes):

            self.e = episode
            # print("\nGame:", episode, end=" ")

            done, score, SAVE = False, 0, ""
            # board = chess.Board() #give whatever starting position here
            #push first move
            result = self.engine.play(self.board,chess.engine.Limit(time=0.01))
            self.board.push(result.move)
            state = self.fen_to_board(self.board.fen())

            moves = 0
            self.info["exploit"], self.info["explore"] = 0, 0
            # self.legal_cnt["illegal"], self.legal_cnt["legal"], self.legal_cnt["NetLegal"] = 0, 0, 0
            while not self.board.is_game_over():

                # if moves % 1000 == 0:
                #     print("{} moves | {} illegal_moves | {} legal moves".format(moves, self.legal_cnt["illegal"], self.legal_cnt["legal"]))

                action = self.getAction(state)

                next_state, reward, done = self.takeAction(action, state)


                if self.broken > 2:
                    self.Model.states, self.Model.actions, self.Model.rewards = None, [], []
                    break
                if done == "stail":
                    self.batch += len(self.Model.rewards)
                    self.trainGame()
                    break
                else:
                    self.Model.memory(state, action, reward)
                    state = next_state
                    score += reward

                if done or self.quickTrain:

                    average = self.Model.PlotModel(score, episode)
                    self.batch += len(self.Model.rewards)

                    if self.max_avg is None or average >= self.max_avg:
                        self.max_avg = average
                        # self.save()
                        # SAVE = "SAVING"
                    else:
                        SAVE = ""
                    
                    if self.quickTrain is False and episode % 40 == 0:
                        print("Game: {}/{}  |  {} Moves in last 5 batches".format(episode, self.episodes, self.batch+len(self.Model.rewards)))
                        print("score: {}, average: {:.2f} {}".format(score, average, SAVE))
                        # print("{} moves | {} illegal_moves | {} legal moves | {} NetLegal moves".format(moves, self.legal_cnt["illegal"], self.legal_cnt["legal"], self.legal_cnt["NetLegal"]))
                        print("{} moves | Exploit: {} | Explore: {}".format(moves, self.info["exploit"], self.info["explore"]))
                        self.batch = 0

                    self.trainGame()
                moves += 1

            self.reset()


        self.engine.quit()


















































                # if states is not None:
                #     # x_samples = np.hstack((x_samples, tools.fen_to_board(board.fen())))
                #     # print("cat")
                #     states = np.concatenate((states, self.fen_to_board(self.board.fen())), axis=0)
                # else:
                #     states = self.fen_to_board(self.board.fen())

                # try:
                #     result = self.engine.play(self.board,chess.engine.Limit(time=0.01))
                #     self.board.push(result.move)
                #     if x_samples is not None:
                #         # x_samples = np.hstack((x_samples, tools.fen_to_board(board.fen())))
                #         # print("cat")
                #         x_samples = np.concatenate((x_samples, self.fen_to_board(self.board.fen())), axis=0)
                #     else:
                #         x_samples = self.fen_to_board(self.board.fen())
                #     try:
                #         result = self.engine.play(self.board,chess.engine.Limit(time=0.1))
                #     except:
                #         x_samples = np.delete(x_samples, x_samples.shape[0] - 1, 0)
                #         # print("broken4")
                #         self.board.reset()
                #         continue
                #     if result.move is None:
                #         x_samples = np.delete(x_samples, x_samples.shape[0] - 1, 0)
                #         # print("broken5")
                #         self.board.reset()
                #         break
                #     labels.append(str(result.move))
                #     self.board.push(result.move)
                # except Exception:
                #     self.board.reset()
