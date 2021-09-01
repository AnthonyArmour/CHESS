import numpy as np
import chess
import chess.engine
from save import save, load
import os
import sys

arguments = sys.argv
gamecount = int(arguments[1])
path = os.getcwd()
engine = chess.engine.SimpleEngine.popen_uci("stockfish_14_linux_x64_avx2/stockfish_14_x64_avx2")

dictsidetomove = {True:'white',False:'black'}
notationdict = {True:'.', False:'...'}

# board.fen()

classes = load("data/classes.pkl")
if classes is None:
    classes = []

for i in range(gamecount):
    board = chess.Board() #give whatever starting position here
    while not board.is_game_over():
        for mv in board.legal_moves:
            classes.append(str(mv))
            board.push(mv)
            for mv2 in board.legal_moves:
                classes.append(str(mv2))
                board.push(mv2)
                for mv3 in board.legal_moves:
                    classes.append(str(mv3))
                board.pop()
            board.pop()

        result = engine.play(board,chess.engine.Limit(time=0.1))
        board.push(result.move)
        classes = list(set(classes))
    print("Games Played:", i)
print("Number of Classes:", len(classes))

k = {}
for x, key in enumerate(classes):
    k[key] = x

save(k, "data/classes.pkl")
print("Saved!")
engine.quit()