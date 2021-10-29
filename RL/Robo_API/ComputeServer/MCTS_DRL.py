import chess
import random
import numpy as np


class Node():
  def __init__(self, turn):
    self.turn = turn
    self.leaf = True
    self.terminal = False
    self.root = False

  def expand(self, moves, priors):
    m = len(moves)
    if m == 0:
      self.terminal = True
    else:
      # S stores the cumulative reward for each of the m moves
      self.Wc = np.zeros(m)
      # Mean action values
      self.Q = np.zeros(m)
      # Prior probabilities given by policy network
      self.Prior_probs = priors
      # T stores the number of plays for each of the m moves
      self.Nt = np.full(m, 0.001)
      # moves stores the list of moves available in this node
      self.moves = moves

      # Change player perspective
      if self.turn is True:
        turn = False
      else:
        turn = True

      self.children = [Node(turn) for a in range(m)]
      self.leaf = False

  def update(self, idx, score):
    self.Wc[idx] += score
    self.Nt[idx] += 1
    self.Q[idx] = self.Wc[idx]/self.Nt[idx]

  def choose(self):
    if self.root is False or random.random() > 0.06:
      choices = 3*self.Prior_probs*(np.sqrt(self.Nt.sum())/1+self.Nt)
      if random.random() < 0.06:
        return np.argmax((choices+self.Q)*np.random.rand(len(choices)))
    else:
      return random.randint(0, len(self.Q)-1)
    return np.argmax(choices+self.Q)



class MCTS():
  def __init__(
    self, graph, IactionSpace, action_space, Imirror, mirror, iterations=5000,
    Actor=None, Value_Net=None, Oponent=None
    ):
    self.Node = None
    self.graph = graph
    self.board = chess.Board()
    self.iterations = iterations
    self.actionSpace = action_space
    self.Imirror = Imirror
    self.mirror = mirror
    self.temp = 1
    self.IactionSpace = IactionSpace
    self.Actor = Actor
    self.Value_Net = Value_Net
    self.Oponent = Oponent

  def act(self, turn=None):
    move, idx = self.search(turn) 
    return move, idx

  def feed(self, move, idx=None):
    self.board.push(move)
    # Moves root node along to prerserve information
    if idx is not None:
      self.Node = self.Node.children[idx]
      self.Node.root = True

  def search(self, turn):
    # print("MCTS searching")

    # create the root note
    if self.Node is None:
      self.Node = Node(turn)
      self.Node.root = True


    for t in range(self.iterations):
      self.mcts(self.Node)

    # print("Total Tree Node Count: ", self.total_node_count())

    # find the index of the most played move at the root node and associated move
    temp = 1/self.temp
    idx = np.argmax((self.Node.Nt**temp)/((self.Node.Nt**temp).sum()))
    move = self.Node.moves[idx]
    # print("\n\nAverage Q Values - MyTurn {} \n{}".format(self.Node.turn, self.Node.Q))
    # for child in self.Node.children:
    #   try:
    #     print("Next nodes Q values: ", child.Q)
    #   except:
    #     pass

    return chess.Move.from_uci(move), idx

  def flip_perspective(self, board):
      return (-np.rot90(board[0, ..., 0], 2))[np.newaxis, ..., np.newaxis]

  def Get_Priors(self, turn):
    legal_moves = [str(mv) for mv in list(self.board.legal_moves)]
    if turn:
      state = self.fen_to_board(self.board.fen())
      with self.graph.as_default():
        prediction = self.Actor.predict(state)[0]
      priors = np.zeros(len(legal_moves))
      for x, mv in enumerate(legal_moves):
        priors[x] = prediction[self.actionSpace[mv]]
      return priors, legal_moves
    else:
      state = self.fen_to_board(self.board.fen())
      state = self.flip_perspective(state)
      with self.graph.as_default():
        prediction = self.Oponent.predict(state)[0]
      priors = np.zeros(len(legal_moves))
      for x, mv in enumerate(legal_moves):
        priors[x] = prediction[self.actionSpace[self.Imirror[mv]]]
      return priors, legal_moves

  def mcts(self, node):
    # if the node is a leaf, then expand it
    if node.leaf:
      priors, moves = self.Get_Priors(node.turn)
      node.expand(moves, priors)
      rollout = True
    else:
      rollout = False

    if node.terminal:
      return 0

    # choose a move 
    idx = node.choose()
    move = node.moves[idx]
    self.board.push(chess.Move.from_uci(move))
    # if the move wins, the value is 1
    if self.board.is_checkmate():
      val = 1
    elif rollout:                 # if we just expanded a node, then get value from rollout
      val = -self.rollout(not node.turn, 0)
    else:
      # otherwise continue traversing the tree
      val = -self.mcts(node.children[idx])
    self.board.pop()

    # update the value of the associated move
    node.update(idx, val)
    return val


  def rollout(self, turn, count):

    if self.board.is_game_over():
        return 0
    move = self.getMove(turn)
    self.board.push(move)

    if self.board.is_checkmate():
        val = 1
    else:
      if turn:
        with self.graph.as_default():
          val = self.Value_Net.predict(self.fen_to_board(self.board.fen()))[0]
        # print("turn {} - value: {}".format(turn, val))
      else:
        state = self.flip_perspective(self.fen_to_board(self.board.fen()))
        with self.graph.as_default():
          val = self.Value_Net.predict(state)[0]
        # print("turn {} - value: {}".format(turn, val))
    # else:
    #     val = -self.rollout(not turn, count+1)
    self.board.pop()
    return val


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


  def getMove(self, turn):
    if turn:
      state = self.fen_to_board(self.board.fen())
      return self.getAction(self.board, state)
    else:
      state = self.flip_perspective(self.fen_to_board(self.board.fen()))
      return self.get_enemy_action(self.board, state)

  def getAction(self, board, state):

      legal_moves = [str(mv) for mv in list(board.legal_moves)]
      with self.graph.as_default():
        prediction = self.Actor.predict(state)[0]
      lm = np.zeros(prediction.shape[0])

      for mv in legal_moves:
          if self.actionSpace[mv] == 0:
              bugValue = True
          lm[self.actionSpace[mv]] = 1
      prediction = prediction.astype(np.float128) + lm.astype(np.float128)

      actions = np.argsort(-prediction, axis=0)[:5]

      for i in actions:
          if self.IactionSpace[i] in legal_moves:
              action = i
              break

      Action = chess.Move.from_uci(self.IactionSpace[action])
      return Action


  def get_enemy_action(self, board, state):
      legal_moves = [str(mv) for mv in list(board.legal_moves)]
      with self.graph.as_default():
        prediction = self.Oponent.predict(state)[0]
      Legal_moves = np.zeros(prediction.shape[0])

      for mv in legal_moves:
          # if self.actionSpace[mv] == 0:
          #     bugValue = True
          legal = self.actionSpace[self.Imirror[mv]]
          Legal_moves[legal] = 1
      prediction = prediction.astype(np.float128) + Legal_moves.astype(np.float128)

      action = np.argmax(prediction)
      Action = chess.Move.from_uci(self.mirror[self.IactionSpace[action]])
      return Action


  def total_node_count(self):
    return self.recurse_nodes(self.Node)


  def recurse_nodes(self, node):
    if node.leaf:
      return 1

    c = 0
    for child in node.children:
      c += self.recurse_nodes(child)
    return c + 1