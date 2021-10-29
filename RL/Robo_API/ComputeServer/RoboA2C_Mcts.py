import os
import pickle
import random
import numpy as np
import chess
from A2C_Model_2 import ActorCritic
import tensorflow as tf
from MCTS_DRL import MCTS


class A3CAgent():

    def __init__(self, graph, lr=0.00001, version=None):
        self.gamma = 0.9
        self.graph = graph
        self.end_game_reward = 0
        self.WHITE = False
        self.BLACK = True
        self.version = version
        self.wins, self.losses = 0.000001, 0.000001

        self.path = os.getcwd()
        self.actionSpace = self.load("../data/classes.pkl")
        self.actionSpaceL = 1968
        mirror = self.load("../data/mirror_classes.pkl")
        IactionSpace = self.load("../data/inverted_classes.pkl")
        Imirror = {}
        for k, v in mirror.items():
            Imirror[v] = k

        if not os.path.exists('../Weights/'): os.makedirs('../Weights/')
        self.paths = {"actor": os.path.join('../Weights/Actor')}

        if not os.path.exists('../Weights/'): os.makedirs('../Weights/')
        self.paths["critic"] = os.path.join('../Weights/Critic')

        if not os.path.exists('../Weights/'): os.makedirs('../Weights/')
        self.paths["coach"] = os.path.join('../Weights/Coach')


        Model = ActorCritic(self.actionSpaceL, lr=lr)

        Actor, Critic, Coach = Model.get_ActorCritic_ResConvNet()
    
        del Model

        self.MCTS = MCTS(
            graph, IactionSpace, self.actionSpace, Imirror, mirror,
            iterations=150, Actor=Actor, Value_Net=Critic, Oponent=Coach
            )
        del Imirror

    def load_weights(self, weights):
        # weights = (Actor, Critic, Coach)
        with self.graph.as_default():
            self.MCTS.Actor.set_weights(weights[0])
            self.MCTS.Oponent.set_weights(weights[2])
            self.MCTS.Value_Net.set_weights(weights[1])


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


    def Run_Games(self, connection, episodes, mcts_iterations, weights=None):
        self.wins, self.losses = 0.0001, 0.0001
        self.MCTS.iterations = mcts_iterations
        if weights is not None:
            self.load_weights(weights)

        for e in range(episodes):

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

        connection.send([state_memory, action_memory, reward_memory, self.wins, self.losses])
        connection.close()


