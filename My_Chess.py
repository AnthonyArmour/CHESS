#!/usr/bin/python3
print ("Content-type:text/html\n")
print ("<h1>This is the Home Page</h1>")
import sys

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'
         

def print_chess_board(board):
    st = ""
    box = "|         " * 8
    box = box + "| "
    st0 = "          " * 8
    x = 0
    #print("\u0332".join("HELLO "))
    print("   ", end="")
    for n in ["A", "B", "C", "D", "E", "F", "G", "H",]:
            print("     {}    ".format(n), end="")
    print()
    print("   ", end="")
    print("\u0332".join(st0 + "  "))
    for n in reversed(range(1, 9)):
        print("   ", end="")
        print(box)
        print(str(n) + "  ", end="")
        for i in ["A", "B", "C", "D", "E", "F", "G", "H",]:
            if x % 2 == 0:
                blank = "        "
            else:
                blank = "\033[40;33m        "
            st = st + str(i) + str(n)
            if board[st] == None:
                print("|{} ".format(blank), end="\033[0m")
            else:
                if x % 2 != 0:
                    print("|", end="")
                    print("\033[40;33m" + "{} ".format(board[st]), end="\033[0m")
                else:
                    print("|{} ".format(board[st]), end="\033[0m")
            st = ""
            x += 1
        x += 1
        print("| ")
        print("   ", end="")
        print("\u0332".join(box))


def player_move(board, pieces, player):
    move_piece = ""
    move_to = ""
    while move_piece not in pieces.keys() or move_to not in board.keys():
        print("Usage: <lowercase_piece> to <uppercase_index>")
        print("Example: knight to F3")
        print("<Case Sensitive>")
        print("Exit: Enter 'exit'\n")
        move = input("{} ".format(player))
        print("\033[0m")
        if move == "exit":
            exit(1)
        move = move.split()
        if len(move) < 3:
            print("1UsageExample: queen to d4")
            continue
        if move[0] not in pieces.keys():
            print("2UsageExample: queen to d4")
            continue
        if move[2] not in board.keys():
            print("3UsageExample: queen to d4")
            continue
        move_piece = move[0]
        move_to = move[2]
    for key in board.keys():
        if board[key] == pieces[move_piece]:
            board[key] = None
    board[move_to] = pieces[move_piece]

def checkwinner(board, p1_king, p2_king):
    p1_safe = False
    p2_safe = False
    for key in board.keys():
        if board[key] == p1_king:
            p1_safe = True
        if board[key] == p2_king:
            p2_safe = True
    if not p2_safe:
        print_chess_board(board)
        print("\033[31m".join("\n\n\n\nPlayer_1 is the winner!\n\n\n\n\n\n"))
        exit(1)
    if not p1_safe:
        print_chess_board(board)
        print("\033[33m".join("\n\n\n\nPlayer_2 is the winner!\n\n\n\n\n\n"))
        exit(1)


def init_chess_board(board, P1_pieces, P2_pieces):
    x = 8
    xx = 1
    for n in ["A", "B", "C", "D", "E", "F", "G", "H",]:
        b = str(n) + "2"
        g = str(n) + "7"
        board[b] = P1_pieces["pawn" + str(xx)]
        board[g] = P2_pieces["pawn" + str(x)]
        x -= 1
        xx += 1
    board["A1"] = P1_pieces["rook1"]
    board["H1"] = P1_pieces["rook2"]
    board["B1"] = P1_pieces["knight1"]
    board["G1"] = P1_pieces["knight2"]
    board["C1"] = P1_pieces["bishop1"]
    board["F1"] = P1_pieces["bishop2"]
    board["D1"] = P1_pieces["king"]
    board["E1"] = P1_pieces["queen"]
    board["H8"] = P2_pieces["rook1"]
    board["A8"] = P2_pieces["rook2"]
    board["G8"] = P2_pieces["knight1"]
    board["B8"] = P2_pieces["knight2"]
    board["F8"] = P2_pieces["bishop1"]
    board["C8"] = P2_pieces["bishop2"]
    board["D8"] = P2_pieces["king"]
    board["E8"] = P2_pieces["queen"]
    return board


P1_pieces = {
    "king": "  King  ",
    "queen": "  Queen ",
    "rook1": "  Rook1 ",
    "rook2": "  Rook2 ",
    "knight1": " Knight1",
    "knight2": " Knight2",
    "bishop1": " Bishop1",
    "bishop2": " Bishop2"
}
P2_pieces = {
    "king": "  King  ",
    "queen": "  Queen ",
    "rook1": "  Rook1 ",
    "rook2": "  Rook2 ",
    "knight1": " Knight1",
    "knight2": " Knight2",
    "bishop1": " Bishop1",
    "bishop2": " Bishop2"
}
for n in range(1, 9):
    key = "pawn" + str(n)
    value = "  Pawn" + str(n) + " "
    P1_pieces[key] = value
    P2_pieces[key] = value

#colorize player_piece dictionaries
for key in P1_pieces.keys():
    P1_pieces[key] = "\033[33m".join(P1_pieces[key])
    P2_pieces[key] = "\033[34m".join(P2_pieces[key])
    print()
#initialize chess board
board = dict()
st = ""
for i in ["A", "B", "C", "D", "E", "F", "G", "H",]:
    for n in range(1, 9):
        st = st + str(i) + str(n)
        board[st] = None
        st = "" 
board = init_chess_board(board, P1_pieces, P2_pieces)
start = input("Want to play Chess?! (yes, no)\n")
while start != "yes" and start != "no":
    print("UsageError: ('yes', 'no')")
    start = input("Want to play Chess?! (yes, no)\n")
if start == "no":
        exit(1)
#BEGIN CHESS GAME
while True:
    print_chess_board(board)
    player_move(board, P1_pieces, "\033[33m".join(" <player_1 move>"))
    checkwinner(board, P1_pieces["king"], P2_pieces["king"])
    print_chess_board(board)
    player_move(board, P2_pieces, "\033[34m".join(" <player_2 move>"))
    checkwinner(board, P1_pieces["king"], P2_pieces["king"])



print_chess_board(board)
#board[st] = "\033[31m".join(" Bishop")