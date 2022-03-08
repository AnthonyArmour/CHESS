def logit(st):
    f = open("/Users/anthonyarmour/VS_Code_Folders/MyGames/GUI_Chess/log_file.txt", "a")
    f.write("\n" + st + "\n")
    f.close()

def log_list_moves(st, lst):
    f = open("/Users/anthonyarmour/VS_Code_Folders/MyGames/GUI_Chess/log_file.txt", "a")

    for item in lst:
        f.write("\n\n\n" + st + item.piece.name + ": " + str(item.move) + "\n\n\n")

    f.close()