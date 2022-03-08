from models.Log_it import logit, log_list_moves
import pygame
import random
import copy
from models.move import Move
from models.Pieces import ChessPiece
from models.Pawn import Pawn
from models.King import King
from models.Knight import Knight
from models.Queen import Queen
from models.Rook import Rook
from models.Bishop import Bishop
"""class structure for cpu that can play chess"""


class RoboChess():
    """Robot chess playing class"""

    def __init__(self, all_pos_lst):
        self.all_pos_list = all_pos_lst


    def robo_move(self, pieces_list, turn):
        blk_lst = []
        wht_lst = []
        for piece in self.my_deepcopy(pieces_list):
            if (piece.team == "white" and turn == "black") or (piece.team == "black" and turn == "white"):
                wht_lst.append(piece)
            else:
                blk_lst.append(piece)

        moves_list = self.find_top_3_moves(blk_lst, wht_lst, True)
        top_moves = []
        for move in moves_list:
            top_moves.append((self.recurse_move(move, 0, self.my_deepcopy(blk_lst), self.my_deepcopy(wht_lst)), move))
        top_moves = sorted(top_moves, key=lambda x: x[0].score, reverse=True)
        for x, move in enumerate(top_moves):
            logit("{}: piece: {} move {}: {}".format(x, move[1].piece.name, move[1].move, move[0].score))
        return self.make_move(top_moves[0][1], pieces_list)

# def __init__(self, name, image, pos, team)
# obj_list.append(Pawn(name="pawn_w1", image=pawn_w, pos=(12, 612), team=w))


    def my_deepcopy(self, lst):
        prior_1 = (True, 2)
        prior_2 = (True, 1)
        prior_3 = (False, 0)
        pawn_p = (False, 0)
        obj_lst = []
        for obj in lst:
            if "pawn" in obj.name:
                if "2" in obj.name or "4" in obj.name or "5" in obj.name or "7" in obj.name:
                    obj_lst.append(Pawn(name=obj.name, image=obj.link, pos=(obj.pos.x, obj.pos.y), team=obj.team, open=(True, 2)))
                else:
                    obj_lst.append(Pawn(name=obj.name, image=obj.link, pos=(obj.pos.x, obj.pos.y), team=obj.team, open=(False, 0)))
            if "bishop" in obj.name:
                obj_lst.append(Bishop(name=obj.name, image=obj.link, pos=(obj.pos.x, obj.pos.y), team=obj.team, open=(False, 0)))
            if "knight" in obj.name:
                obj_lst.append(Knight(name=obj.name, image=obj.link, pos=(obj.pos.x, obj.pos.y), team=obj.team, open=(True, 2)))
            if "rook" in obj.name:
                obj_lst.append(Rook(name=obj.name, image=obj.link, pos=(obj.pos.x, obj.pos.y), team=obj.team, open=(False, 0)))
            if "queen" in obj.name:
                obj_lst.append(Queen(name=obj.name, image=obj.link, pos=(obj.pos.x, obj.pos.y), team=obj.team, open=(False, 0)))
            if "king" in obj.name:
                obj_lst.append(King(name=obj.name, image=obj.link, pos=(obj.pos.x, obj.pos.y), team=obj.team, open=(False, 0)))
        return obj_lst


    def find_top_3_moves(self, active_team_lst, enemy_lst, log):
        moves_list, safe_spots_lst = self.defense(active_team_lst, enemy_lst)
        # if log:
            # log_list_moves("Moves list after Defense: ", moves_list)
        if len(moves_list) < 3:
            moves_list += self.offense(safe_spots_lst, active_team_lst, enemy_lst)
            # if log:
                # log_list_moves("Moves list after Offense: ", moves_list)
        while len(moves_list) < 3:
            moves_list.append(self.find_decent_move(active_team_lst, enemy_lst, safe_spots_lst))
            # if log:
                # log_list_moves("Moves list after find_decent_move: ", moves_list)
        return moves_list[:3]


    def defense(self, active_team_lst, enemy_lst):
        threatened_pcs_lst = self.all_threatened_pcs(active_team_lst, enemy_lst)
        safe_spots = self.all_safe_spots(enemy_lst, active_team_lst)
        if not threatened_pcs_lst:
            return [], safe_spots
        
        threatened_piece = self.determine_pc_to_defend(threatened_pcs_lst)

        valid_moves = []
        for spot in threatened_piece.all_valid_moves(threatened_piece.pos.x, threatened_piece.pos.y, active_team_lst + enemy_lst):
            if spot in safe_spots:
                enemy = self.check_for_enemy(enemy_lst, spot)
                if enemy:
                    value = (threatened_piece.value * 3) + enemy.value # + 200
                else:
                    value = threatened_piece.value * 3 # + 100
                valid_moves.append(Move(spot, value, threatened_piece))
        valid_moves = sorted(valid_moves, key=lambda x: x.score, reverse=True)
        if len(valid_moves) > 2:
            return valid_moves[:2], safe_spots
        return valid_moves, safe_spots


    def opening_moves(self, piece, lst_pieces):
        open_lst = []
        for item in random.sample(lst_pieces, len(lst_pieces)):
            if item.open == (True, 2):
                open_lst = [item] + open_lst
            elif item.open == (True, 1):
                open_lst.append(item)
        if len(open_lst) == 0:
            return False
        elif piece == open_lst[0]:
            return True
        return False


    def check_for_enemy(self, enemy_lst, spot):
        for enemy in enemy_lst:
            if (enemy.pos.x, enemy.pos.y) == spot:
                return enemy
        return None



    def get_enemy_cordinates(self, enemy_lst):
        enemy_cordinates = []
        for piece in enemy_lst:
            enemy_cordinates.append((piece.pos.x, piece.pos.y))
        return enemy_cordinates


    def offense(self, safe_spots_lst, active_team_lst, enemy_lst):

        best_attacks = []

        for blk_piece in active_team_lst:
            spots = blk_piece.all_valid_moves(blk_piece.pos.x, blk_piece.pos.y, enemy_lst + active_team_lst)
            for wht_piece in enemy_lst:
                pos = (wht_piece.pos.x, wht_piece.pos.y)
                if pos in spots and self.weigh_attack(wht_piece, blk_piece, safe_spots_lst, enemy_lst, active_team_lst) is True:
                    move = Move((wht_piece.pos.x, wht_piece.pos.y), wht_piece.value, blk_piece)
                    # if self.opening_moves(move.piece, active_team_lst) is True:
                    #     move.score += 100
                    move.score += wht_piece.value * 2
                    best_attacks.append(move)
        best_attacks = sorted(best_attacks, key=lambda x: x.score, reverse=True)
        return best_attacks


    def weigh_attack(self, enemy, attack_piece, safe_spots, enemy_lst, active_lst):
        
        temp_pos = (attack_piece.pos.x, attack_piece.pos.y)
        temp_enemy_lst = []
        for tmp_enemy in enemy_lst:
            if tmp_enemy.id != enemy.id:
                temp_enemy_lst.append(tmp_enemy)
        for piece in active_lst:
            if attack_piece.id == piece.id:
                piece.pos.x, piece.pos.y = enemy.pos.x, enemy.pos.y
        threatened_pcs_lst = self.all_threatened_pcs(active_lst, enemy_lst)
        for piece in active_lst:
            if attack_piece.id == piece.id:
                piece.pos.x, piece.pos.y = temp_pos[0], temp_pos[1]
        if threatened_pcs_lst:
            for piece in threatened_pcs_lst:
                if piece[0].value >= enemy.value:
                    return False
        if (enemy.pos.x, enemy.pos.y) not in safe_spots:
            if enemy.value > attack_piece.value:
                return True
            else:
                return False
        return True

    def find_decent_move(self, active_team_lst, enemy_lst, safe_spots_lst):

        true_lst = []
        false_lst = []
        ordered_pieces = []
        # sort_lst = []
        # opening = False
        for piece in active_team_lst:
            if piece.name == "king_w" or piece.name == "king_b":
                true_lst.append(piece)
            if piece.moved is False:
                false_lst .append(piece)
            else:
                true_lst.append(piece)

        ordered_pieces = random.sample(false_lst, len(false_lst)) + true_lst
        potential_attack_decisions = []
        non_attack_decisions = []

        for piece in ordered_pieces:
            valid_moves = piece.all_valid_moves(piece.pos.x, piece.pos.y, enemy_lst + active_team_lst)
            temp_potential_attacks = self.set_up_attack(piece, valid_moves, self.get_enemy_cordinates(enemy_lst), active_team_lst + enemy_lst, safe_spots_lst)
            if len(temp_potential_attacks) > 0:
                potential_attack_decisions.append((temp_potential_attacks, piece))
            elif len(valid_moves) > 0:
                non_attack_decisions.append((valid_moves, piece))

        moves = []
        if len(potential_attack_decisions) > 0:
            decision = random.choice(potential_attack_decisions)
        else:
            decision = random.choice(non_attack_decisions)
        for move in decision[0]:
            # if move in safe_spots_lst:
            moves.append(move)
        if len(moves) > 0:
            move = Move(random.choice(moves), 0, decision[1])
            return move

        


    def set_up_attack(self, piece, valid_moves, enemy_cordinates, pieces, safe_spots):
        potential_attacks = []
        for move in valid_moves:
            if move in safe_spots:
                temp_x, temp_y = piece.pos.x, piece.pos.y
                piece.pos.x, piece.pos.y = move[0], move[1]
                self_valid_moves = piece.all_valid_moves(piece.pos.x, piece.pos.y, pieces)
                piece.pos.x, piece.pos.y = temp_x, temp_y
                for mv in self_valid_moves:
                    if mv in enemy_cordinates and mv in safe_spots:
                        potential_attacks.append(move)
        return potential_attacks




    def recurse_move(self, move, idx, blk_lst, wht_lst):
        last_move, tmp_blk_lst, tmp_wht_lst = self.temp_move_score(move, self.my_deepcopy(blk_lst), self.my_deepcopy(wht_lst), idx + 1)
        if idx == 1:
            return last_move

        moves_list = self.find_top_3_moves(tmp_blk_lst, tmp_wht_lst, False)
        top_moves = []
        for mv in moves_list:
            top_moves.append(self.recurse_move(mv, idx + 1, tmp_blk_lst, tmp_wht_lst))

        top_moves = sorted(top_moves, key=lambda x: x.score, reverse=True)
        top_moves[0].score += last_move.score
        return top_moves[0]


    def temp_move_score(self, move, active_team_lst, enemy_lst, idx):
        all_valid_enemy_moves = []
        for x, piece in enumerate(enemy_lst):
            if move.move == (piece.pos.x, piece.pos.y):
                move.score += piece.value * 2
                enemy_lst.pop(x)
        for piece in active_team_lst:
            if move.piece.id == piece.id:
                # piece.pos.x = move.piece.pos.x
                # piece.pos.y = piece.pos.y
                piece.pos.x = move.move[0]
                piece.pos.y = move.move[1]
        
        # if idx < 3:
        #     top_3_moves = self.find_top_3_moves(enemy_lst, active_team_lst, False)
        #     for move in top_3_moves:
        #         all_valid_enemy_moves.append((self.recurse_move(move, 2, enemy_lst, active_team_lst)).move)
        # else:
        enemy_safe_spots = self.all_safe_spots(active_team_lst, enemy_lst)
        for move in self.offense(enemy_safe_spots, enemy_lst, active_team_lst):
            all_valid_enemy_moves.append(move.move)


        worst_attack = 0
        for piece in active_team_lst:
            if (piece.pos.x, piece.pos.y) in all_valid_enemy_moves and piece.value > worst_attack:
                worst_attack = piece.value * 3
                if "king" in piece.name:
                    worst_attack += 15
                if "queen" in piece.name:
                    worst_attack += 10
                if "rook" in piece.name:
                    worst_attack += 5
        move.score -= worst_attack
        return move, active_team_lst, enemy_lst


    def make_move(self, piece_and_move, pieces_list):
        for x, piece in enumerate(pieces_list):
            if piece.name == piece_and_move.piece.name and piece.team == piece_and_move.piece.team:
                piece.prev_pos = (piece_and_move.piece.pos.x, piece_and_move.piece.pos.y)
                piece.pos.x = piece_and_move.move[0]
                piece.pos.y = piece_and_move.move[1]
                # piece.open = (False, 0)
                logit("make move: " + str(piece_and_move.move) + " - " + str(piece.pos.x) + ", " + str(piece.pos.y))
            if (piece.pos.x, piece.pos.y) == piece_and_move.move and piece.team != piece_and_move.piece.team:
                pieces_list.pop(x)
        return pieces_list


    def all_safe_spots(self, enenmy_lst, active_team_lst):
        """returns list of all safe positions on board"""
        enemy_attack_range = []
        safe_spots = []
        for enemy in enenmy_lst:
            enemy_attack_range += enemy.all_valid_moves(enemy.pos.x, enemy.pos.y, enenmy_lst + active_team_lst)
        for pos in self.all_pos_list:
            if pos not in enemy_attack_range:
                safe_spots.append(pos)
        return safe_spots

    
    def determine_pc_to_defend(self, threatened_pcs_lst):
        """returns most valuable piece that is threatened"""
        highest = 0
        ret = None
        for tup in threatened_pcs_lst:
            if tup[0].value >= highest:
                ret = tup[0]
                highest = tup[0].value
        return ret


    def all_threatened_pcs(self, active_team_lst, enemy_lst):
        """return a list of tuples containing a piece and a list of all threats"""
        threatened_pcs_lst = []
        threats_list = []
        for active_piece in active_team_lst:
            for enemy in enemy_lst:
                if (active_piece.pos.x, active_piece.pos.y) in enemy.all_valid_moves(enemy.pos.x, enemy.pos.y, active_team_lst + enemy_lst):
                    threats_list.append(enemy)
            if len(threats_list) > 0:
                threatened_pcs_lst.append((active_piece, list(threats_list)))
                threats_list.clear()
        if len(threatened_pcs_lst) > 0:
            return threatened_pcs_lst
        return None