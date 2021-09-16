import chess
import chess.engine
# import random
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
        self.engine = chess.engine.SimpleEngine.popen_uci("stockfish_14_linux_x64_avx2/stockfish_14_x64_avx2")
        self.path = os.getcwd()
        self.actionSpace = self.load("data/classes.pkl")
        self.IactionSpace = self.load("data/inverted_classes.pkl")
        self.Model = ActorCritic(len(self.actionSpace))
        self.board = chess.Board()
        self.illegal_moves = 0
        self.quickTrain = False
        self.legal_cnt = {
            "illegal": 0,
            "legal": 0
        }

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
        action = chess.Move.from_uci(self.IactionSpace[action])
        if action in self.board.legal_moves:
            self.board.push(action)
            if self.board.is_checkmate():
                reward = 200
                done = True
                next_state = None
            elif self.board.outcome() is not None:
                done = "stail"
            else:
                result = self.engine.play(self.board,chess.engine.Limit(time=0.1))
                self.board.push(result.move)
                next_state = self.fen_to_board(self.board.fen())
                reward = np.sum(next_state) - 64
                if self.board.is_checkmate():
                    reward -= 200
                    done = True
                    next_state = None
            self.legal_cnt["legal"] += 1
        else:
            reward = -50
            next_state = False
            self.illegal_moves += 1
            self.legal_cnt["illegal"] += 1
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
        # reset training memory


    def RunTrainingSession(self):
        # FixState = False

        for episode in range(self.episodes):

            print("\nGame:", episode)

            done, score, SAVE = False, 0, ""
            # board = chess.Board() #give whatever starting position here
            #push first move
            result = self.engine.play(self.board,chess.engine.Limit(time=0.01))
            self.board.push(result.move)
            state = self.fen_to_board(self.board.fen())

            moves = 0
            self.legal_cnt["illegal"], self.legal_cnt["legal"] = 0, 0
            while not self.board.is_game_over():

                # if moves % 200 == 0:
                #     print("{} moves | {} illegal_moves | {} legal moves".format(moves, self.legal_cnt["illegal"], self.legal_cnt["legal"]))

                # if not FixState:
                action = self.Model.getAction(state)

                next_state, reward, done = self.takeAction(action, state)
                # if not next_state:
                #     FixState = True
                #     continue
                # else:
                #     FixState = False

                if done != "stail":
                    self.Model.memory(state, action, reward)
                    state = next_state
                    score += reward

                if done or self.quickTrain:
                    average = self.Model.PlotModel(score, episode)

                    if self.max_avg is None or average >= self.max_avg:
                        self.max_avg = average
                        # self.save()
                        # SAVE = "SAVING"
                    else:
                        SAVE = ""
                    if not self.quickTrain or moves % 1000 == 0:
                        print("episode: {}/{}, score: {}, average: {:.2f} {}".format(episode, self.episodes, score, average, SAVE))
                        print("{} moves | {} illegal_moves | {} legal moves".format(moves, self.legal_cnt["illegal"], self.legal_cnt["legal"]))

                    self.trainGame()
                moves += 1

            self.board.reset()


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