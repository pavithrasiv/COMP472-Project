from __future__ import annotations
import argparse
import copy
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from time import sleep
from tkinter import CURRENT
from typing import Tuple, TypeVar, Type, Iterable, ClassVar
import random
import requests
import logging

# maximum and minimum values for our heuristic scores (usually represents an end of game condition)
MAX_HEURISTIC_SCORE = 2000000000
MIN_HEURISTIC_SCORE = -2000000000


class UnitType(Enum):
    """Every unit type."""
    AI = 0
    Tech = 1
    Virus = 2
    Program = 3
    Firewall = 4

class Player(Enum):
    """The 2 players."""
    Attacker = 0
    Defender = 1

    def next(self) -> Player:
        """The next (other) player."""
        if self is Player.Attacker:
            return Player.Defender
        else:
            return Player.Attacker

class GameType(Enum):
    AttackerVsDefender = 0
    AttackerVsComp = 1
    CompVsDefender = 2
    CompVsComp = 3

##############################################################################################################

@dataclass(slots=True)
class Unit:
    player: Player = Player.Attacker
    type: UnitType = UnitType.Program
    health : int = 9
    # class variable: damage table for units (based on the unit type constants in order)
    damage_table : ClassVar[list[list[int]]] = [
        [3,3,3,3,1], # AI
        [1,1,6,1,1], # Tech
        [9,6,1,6,1], # Virus
        [3,3,3,3,1], # Program
        [1,1,1,1,1], # Firewall
    ]
    # class variable: repair table for units (based on the unit type constants in order)
    repair_table : ClassVar[list[list[int]]] = [
        [0,1,1,0,0], # AI
        [3,0,0,3,3], # Tech
        [0,0,0,0,0], # Virus
        [0,0,0,0,0], # Program
        [0,0,0,0,0], # Firewall
    ]

    def is_alive(self) -> bool:
        """Are we alive ?"""
        return self.health > 0

    def mod_health(self, health_delta : int):
        """Modify this unit's health by delta amount."""
        self.health += health_delta
        if self.health < 0:
            self.health = 0
        elif self.health > 9:
            self.health = 9

    def to_string(self) -> str:
        """Text representation of this unit."""
        p = self.player.name.lower()[0]
        t = self.type.name.upper()[0]
        return f"{p}{t}{self.health}"
    
    def __str__(self) -> str:
        """Text representation of this unit."""
        return self.to_string()
    
    def damage_amount(self, target: Unit) -> int:
        """How much can this unit damage another unit."""
        amount = self.damage_table[self.type.value][target.type.value]
        if target.health - amount < 0:
            return target.health
        return amount

    def repair_amount(self, target: Unit) -> int:
        """How much can this unit repair another unit."""
        amount = self.repair_table[self.type.value][target.type.value]
        if target.health + amount > 9:
            return 9 - target.health
        return amount
    
    def self_destruct(self):
        """Reduce health of unit to self-destruct"""
        self.health = 0


##############################################################################################################

@dataclass(slots=True)
class Coord:
    """Representation of a game cell coordinate (row, col)."""
    row : int = 0
    col : int = 0

    def col_string(self) -> str:
        """Text representation of this Coord's column."""
        coord_char = '?'
        if self.col < 16:
                coord_char = "0123456789abcdef"[self.col]
        return str(coord_char)

    def row_string(self) -> str:
        """Text representation of this Coord's row."""
        coord_char = '?'
        if self.row < 26:
                coord_char = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[self.row]
        return str(coord_char)

    def to_string(self) -> str:
        """Text representation of this Coord."""
        return self.row_string()+self.col_string()
    
    def __str__(self) -> str:
        """Text representation of this Coord."""
        return self.to_string()
    
    def clone(self) -> Coord:
        """Clone a Coord."""
        return copy.copy(self)

    def iter_range(self, dist: int) -> Iterable[Coord]:
        """Iterates over Coords inside a rectangle centered on our Coord."""
        for row in range(self.row-dist,self.row+1+dist):
            for col in range(self.col-dist,self.col+1+dist):
                yield Coord(row,col)

    def iter_adjacent(self) -> Iterable[Coord]:
        """Iterates over adjacent Coords."""
        yield Coord(self.row-1,self.col)
        yield Coord(self.row,self.col-1)
        yield Coord(self.row+1,self.col)
        yield Coord(self.row,self.col+1)

    @classmethod
    def from_string(cls, s : str) -> Coord | None:
        """Create a Coord from a string. ex: D2."""
        s = s.strip()
        for sep in " ,.:;-_":
                s = s.replace(sep, "")
        if (len(s) == 2):
            coord = Coord()
            coord.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[0:1].upper())
            coord.col = "0123456789abcdef".find(s[1:2].lower())
            return coord
        else:
            return None

##############################################################################################################

@dataclass(slots=True)
class CoordPair:
    """Representation of a game move or a rectangular area via 2 Coords."""
    src : Coord = field(default_factory=Coord)
    dst : Coord = field(default_factory=Coord)

    def to_string(self) -> str:
        """Text representation of a CoordPair."""
        return self.src.to_string()+" "+self.dst.to_string()
    
    def __str__(self) -> str:
        """Text representation of a CoordPair."""
        return self.to_string()

    def clone(self) -> CoordPair:
        """Clones a CoordPair."""
        return copy.copy(self)

    def iter_rectangle(self) -> Iterable[Coord]:
        """Iterates over cells of a rectangular area."""
        for row in range(self.src.row,self.dst.row+1):
            for col in range(self.src.col,self.dst.col+1):
                yield Coord(row,col)

    @classmethod
    def from_quad(cls, row0: int, col0: int, row1: int, col1: int) -> CoordPair:
        """Create a CoordPair from 4 integers."""
        return CoordPair(Coord(row0,col0),Coord(row1,col1))
    
    @classmethod
    def from_dim(cls, dim: int) -> CoordPair:
        """Create a CoordPair based on a dim-sized rectangle."""
        return CoordPair(Coord(0,0),Coord(dim-1,dim-1))
    
    @classmethod
    def from_string(cls, s : str) -> CoordPair | None:
        """Create a CoordPair from a string. ex: A3 B2"""
        s = s.strip()
        for sep in " ,.:;-_":
                s = s.replace(sep, "")
        if (len(s) == 4):
            coords = CoordPair()
            coords.src.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[0:1].upper())
            coords.src.col = "0123456789abcdef".find(s[1:2].lower())
            coords.dst.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[2:3].upper())
            coords.dst.col = "0123456789abcdef".find(s[3:4].lower())
            return coords
        else:
            return None

##############################################################################################################

@dataclass(slots=True)
class Options:
    """Representation of the game options."""
    dim: int = 5
    max_depth : int | None = 4
    min_depth : int | None = 2
    max_time : float | None = None
    game_type : GameType = None
    alpha_beta : bool = None
    max_turns : int | None = None
    randomize_moves : bool = True
    broker : str | None = None
    heuristic: int | None = 0

##############################################################################################################

@dataclass(slots=True)
class Stats:
    """Representation of the global game statistics."""
    evaluations_per_depth : dict[int,int] = field(default_factory=dict)
    total_seconds: float = 0.0

##############################################################################################################

@dataclass(slots=True)
class Game:
    """Representation of the game state."""
    board: list[list[Unit | None]] = field(default_factory=list)
    next_player: Player = Player.Attacker
    turns_played : int = 0
    options: Options = field(default_factory=Options)
    stats: Stats = field(default_factory=Stats)
    _attacker_has_ai : bool = True
    _defender_has_ai : bool = True
    total_evals: int = 0
    evals_by_depth: list[int] = field(default_factory=list)

    def __post_init__(self):
        """Automatically called after class init to set up the default board state."""
        dim = self.options.dim
        self.board = [[None for _ in range(dim)] for _ in range(dim)]
        md = dim-1
        self.set(Coord(0,0),Unit(player=Player.Defender,type=UnitType.AI))
        self.set(Coord(1,0),Unit(player=Player.Defender,type=UnitType.Tech))
        self.set(Coord(0,1),Unit(player=Player.Defender,type=UnitType.Tech))
        self.set(Coord(2,0),Unit(player=Player.Defender,type=UnitType.Firewall))
        self.set(Coord(0,2),Unit(player=Player.Defender,type=UnitType.Firewall))
        self.set(Coord(1,1),Unit(player=Player.Defender,type=UnitType.Program))
        self.set(Coord(md,md),Unit(player=Player.Attacker,type=UnitType.AI))
        self.set(Coord(md-1,md),Unit(player=Player.Attacker,type=UnitType.Virus))
        self.set(Coord(md,md-1),Unit(player=Player.Attacker,type=UnitType.Virus))
        self.set(Coord(md-2,md),Unit(player=Player.Attacker,type=UnitType.Program))
        self.set(Coord(md,md-2),Unit(player=Player.Attacker,type=UnitType.Program))
        self.set(Coord(md-1,md-1),Unit(player=Player.Attacker,type=UnitType.Firewall))

        self.total_evals = 0

    def print_board(self):
        board_output = self.to_string()
        print(board_output)



    def clone(self) -> Game:
        """Make a new copy of a game for minimax recursion.

        Shallow copy of everything except the board (options and stats are shared).
        """
        new = copy.copy(self)
        new.board = copy.deepcopy(self.board)
        return new

    def is_empty(self, coord : Coord) -> bool:
        """Check if contents of a board cell of the game at Coord is empty (must be valid coord)."""
        return self.board[coord.row][coord.col] is None


    def get(self, coord : Coord) -> Unit | None:
        """Get contents of a board cell of the game at Coord."""
        if self.is_valid_coord(coord):
            return self.board[coord.row][coord.col]
        else:
            return None

    def set(self, coord : Coord, unit : Unit | None):
        """Set contents of a board cell of the game at Coord."""
        if self.is_valid_coord(coord):
            self.board[coord.row][coord.col] = unit

    def heuristic_e2(self) -> int:
        # implementation of heuristic to check the defense stability of the program 
        # e(n) = (health of the F and P of Player 1) - (health of the F and P of Player 2)
        attackerHealth = 0  
        defenderHealth = 0

        for coord in CoordPair.from_dim(self.options.dim).iter_rectangle():
            unit = self.get(coord)
            if unit is None:  # Guard clause to check if unit is None.
                continue  # Skip to next iteration if unit is None.
            
            if unit.type != UnitType.AI:
                if unit.type != UnitType.Virus:
                    if unit.type != UnitType.Tech:
                        # attacker
                        if unit.player is Player.Attacker:
                            # add firewall health of attacker
                            if unit.type is UnitType.Firewall:
                                attackerHealth = attackerHealth + unit.health 
                            # add program health of attacker
                            elif unit.type is UnitType.Program:
                                attackerHealth = attackerHealth + unit.health 
                        # defender 
                        elif unit.player is Player.Defender:
                            # add firewall health of defender
                            if unit.type is UnitType.Firewall:
                                defenderHealth = defenderHealth + unit.health
                            # add program health of defender
                            elif unit.type is UnitType.Program:
                                defenderHealth = defenderHealth + unit.health

        # perform the heuristics 
        e2 = attackerHealth - defenderHealth
        return e2

    def remove_dead(self, coord: Coord):
        """Remove unit at Coord if dead."""
        unit = self.get(coord)
        if unit is not None and not unit.is_alive():
            self.set(coord,None)
            if unit.type == UnitType.AI:
                if unit.player == Player.Attacker:
                    self._attacker_has_ai = False
                else:
                    self._defender_has_ai = False

    def mod_health(self, coord : Coord, health_delta : int):
        """Modify health of unit at Coord (positive or negative delta)."""
        target = self.get(coord)
        if target is not None:
            target.mod_health(health_delta)
            self.remove_dead(coord)

    def is_valid_move(self, coords : CoordPair) -> bool:
        """Validate a move expressed as a CoordPair. TODO: WRITE MISSING CODE!!!"""
        if not self.is_valid_coord(coords.src) or not self.is_valid_coord(coords.dst):
            return False
        unit = self.get(coords.src)
        unit_dst = self.get(coords.dst)
        if unit is None or unit.player != self.next_player:
            return False
       
        #check if unit is trying to self-destruct
        if coords.src == coords.dst:
            return True

        #check unit type
        adjacent_coords = list(coords.src.iter_adjacent())

        row_decrement = adjacent_coords[0]
        column_decrement = adjacent_coords[1]
        row_increment = adjacent_coords[2]
        column_increment = adjacent_coords[3]

        unit = self.get(coords.src)
        if self.in_repair(coords) is True:
            return True 

        
        if self.in_combat(coords) is True:
            if unit_dst != self.next_player:
                return True
            else:
                return False
        if not self.in_combat(coords) or not self.in_repair(coords):
            if unit_dst is not None:
                    return False
            if unit.type != UnitType.Virus and unit.type != UnitType.Tech:
                if unit.player is Player.Attacker:
                    if coords.dst != row_decrement and coords.dst != column_decrement:
                        return False
                if unit.player is Player.Defender:
                    if coords.dst != row_increment and coords.dst != column_increment:
                        return False    
        return True
    
    def in_repair(self, coords):
        #check the destination and and if in the same team
        if self.is_empty(coords.dst):
            return False 
        if coords.dst == coords.src:
            return False
        else:
            unit = self.get(coords.dst)
            if unit.player is self.next_player:
                if  unit.health == 9:
                    return False
                else:
                    return True
            else: 
                return False
            

    def in_combat(self, coords) -> bool:
        """ Checks adjacent cells for enemy units. If present, unit is in combat"""
        adjacent_coords = list(coords.src.iter_adjacent())

        row_decrement = adjacent_coords[0]
        column_decrement = adjacent_coords[1]
        row_increment = adjacent_coords[2]
        column_increment = adjacent_coords[3]


    # if the cell is not empty, the unit is in combat
        
        if self.is_valid_coord(row_increment):
            if not self.is_empty(row_increment):
                unit = self.get(row_increment)
                if unit.player is not self.next_player:
                    return True
        if self.is_valid_coord(row_decrement):
            if not self.is_empty(row_decrement) and self.is_valid_coord(row_decrement):
                unit = self.get(row_decrement)
                if unit.player is not self.next_player:
                    return True
        if self.is_valid_coord(column_increment):
            if not self.is_empty(column_increment) and self.is_valid_coord(column_increment):
                unit = self.get(column_increment)
                if unit.player is not self.next_player:
                    return True
        if self.is_valid_coord(column_decrement):
            if not self.is_empty(column_decrement) and self.is_valid_coord(column_decrement):
                unit = self.get(column_decrement)
                if unit.player is not self.next_player:
                    return True
        
        return False
        

    def perform_move(self, coords : CoordPair, calledFromMinimax) -> Tuple[bool,str]:
        """Validate and perform a move expressed as a CoordPair. TODO: WRITE MISSING CODE!!!"""
        if self.is_valid_move(coords):

            if self.in_repair(coords):
            #check if its not the next player, want to repair
                repair = self.unit_repair(coords)
                if not calledFromMinimax:
                    logging.info('---Action Information---')
                    logging.info(f'Turn #{self.turns_played+1}')
                    logging.info(f'Unit at {coords.src.to_string()} repaired unit at {coords.dst.to_string()} by {self.next_player}\n')

            #check that the source unit is not empty
            if self.get(coords.src) is not None:
                #check that the target unit is not empty
                if self.get(coords.dst) is not None:
                    unit = self.get(coords.dst)
                    if unit.player is not self.next_player:
                        damage = self.get(coords.src).damage_amount(self.get(coords.dst))
                        #Decrease health of the source unit according to the damage table
                        self.mod_health(coords.src, -damage)
                        #Decrease health of the target unit according to the damage table
                        self.mod_health(coords.dst, -damage)
                        if not calledFromMinimax:
                            logging.info('---Action Information---\n')
                            logging.info(f'Turn #{self.turns_played+1}')
                            logging.info(f'Unit at {coords.src.to_string()} attacked unit at {coords.dst.to_string()} by {self.next_player}\n')
                    elif coords.src == coords.dst:
                        self.unit_self_destruct(coords)
                        self.set(coords.src, None)
                        if not calledFromMinimax:
                            logging.info('---Action Information---\n')
                            logging.info(f'Turn #{self.turns_played+1}')
                            logging.info(f'Unit at {coords.src.to_string()} self destructed by {self.next_player}\n')

                else:
                    #if there is no unit at the destination, move the source to that coordinate
                    self.set(coords.dst,self.get(coords.src))
                    self.set(coords.src,None)
                    if not calledFromMinimax:
                        logging.info('---Action Information---\n')
                        logging.info(f'Turn #{self.turns_played+1}')
                        logging.info(f'move from {coords.src.to_string()} to {coords.dst.to_string()} by {self.next_player}\n')
            if calledFromMinimax:
                self.next_turn()
            return (True,"")
        return (False,"invalid move")

    #repare when it is going to be an issue
    def unit_repair(self, coords: CoordPair) -> str:
         """Allow a unit to self repair"""
         unit = self.get(coords.src)
         unit_repairing = self.get(coords.src).repair_amount(self.get(coords.dst))

         if unit is not None:
             
            #repair amount based on the repair values 
            #repair_amount = unit.repair_amount(units_repair)
            #if the unit has 0 lives, it is dead 
            if unit_repairing == 0: 
                return "Invalid repair, no lives"
                #if the unit has all their lives, doesnt need to be repaired 
            elif unit_repairing == 9: 
                return "Invalid repair, has all their lives"
            else:
                #want to set the repair amount to the destination unit
                self.get(coords.dst).mod_health(+unit_repairing)
             #self.set(coords.src, None)
            return "self-repaired successful"
         return "Self-repaired failed."

    def unit_self_destruct(self, coords: CoordPair) -> str:
        """Allow a unit to self destruct"""
        unit = self.get(coords.src)

        if unit is not None:
            #Damage health of 4 diagonal surrounding units from the self-destructing unit
            diagonal_units = list(coords.src.iter_range(1))
            for j in diagonal_units:
                units_diag = self.get(j)
                if units_diag is not None:
                    units_diag.mod_health(-2)

            unit.self_destruct()
            self.remove_dead(coords.src)
            return "Self-destruction occurred."
        return "Self-destruction failed."
    

    # Heuristic function e0 (given)
    # Heuristic that uses the number of unit for each type
    def heuristic_e0(self) -> int:
        Vp1, Tp1, Fp1, Pp1, AIp1 = 0, 0, 0, 0, 0
        Vp2, Tp2, Fp2, Pp2, AIp2 = 0, 0, 0, 0, 0
        
        for coord in CoordPair.from_dim(self.options.dim).iter_rectangle():
                unit = self.get(coord)
                if unit is not None:
                    if unit.player is Player.Attacker:
                        if unit.type == UnitType.Virus:
                            Vp1 += 1
                        if unit.type == UnitType.Tech:
                            Tp1 += 1
                        if unit.type == UnitType.Firewall:
                            Fp1 += 1
                        if unit.type == UnitType.Program:
                            Pp1 += 1
                        if unit.type == UnitType.AI:
                            AIp1 += 1

                    elif unit.player is Player.Defender:
                        if unit.type == UnitType.Virus:
                            Vp2 += 1
                        if unit.type == UnitType.Tech:
                            Tp2 += 1
                        if unit.type == UnitType.Firewall:
                            Fp2 += 1
                        if unit.type == UnitType.Program:
                            Pp2 += 1
                        if unit.type == UnitType.AI:
                            AIp2 += 1
                        

        e0 = (3*Vp1 + 3*Tp1 + 3*Fp1 + 3*Pp1 + 9999*AIp1) - (3*Vp2 + 3*Tp2 + 3*Fp2 + 3*Pp2 + 9999*AIp2)

        return e0

    def heuristic_e1(self) -> float:
        """ Heuristic which uses the units' current health to assess the game state """
        attackerHealth = 0
        defenderHealth = 0

        for coord in CoordPair.from_dim(self.options.dim).iter_rectangle():
            unit = self.get(coord)
            if unit is not None:
                if unit.type != UnitType.AI:
                    if unit.player is Player.Attacker:
                        attackerHealth += unit.health * 3
                    elif unit.player is Player.Defender:
                        defenderHealth += unit.health * 3
                else:
                    if unit.player is Player.Attacker:
                        attackerHealth += unit.health * 9999
                    elif unit.player is Player.Defender:
                        defenderHealth += unit.health * 9999
        e1 = attackerHealth - defenderHealth
        return e1

    def minimax_withab(self, depth, max_player, alpha, beta):
        """ Function for determining best move for AI to make using minimax algorithm and alpha beta pruning """
        start_time = datetime.now()
        if depth == 0 or self.has_winner() is not None:
            if self.options.heuristic == 0:
                return self.heuristic_e0(), None, depth
            if self.options.heuristic == 1:
                return self.heuristic_e1(), None, depth
            if self.options.heuristic == 2:
                return self.heuristic_e2(), None, depth

        if max_player:
            max_eval = -float('inf')
            best_move = None
            count = 0
            for move in self.move_candidates():
                elapsed_seconds = (datetime.now() - start_time).total_seconds()
                # Check if there is still time to run algorithm
                if elapsed_seconds > self.options.max_time:
                    print("AI ran out of time")
                    logging.info(f'AI ran out of time')
                    break
                self_clone = self.clone()
                self_clone.perform_move(move, True)
                eval = self_clone.minimax_withab(depth - 1, False, alpha, beta)
                if eval[0] > max_eval:
                    max_eval = eval[0]
                    best_move = move
                alpha = max(alpha, eval[0])
                if beta <= alpha:
                    break # Beta cutoff
                count += 1
            return max_eval, best_move, count
        else:
            min_eval = float('inf')
            best_move = None
            count = 0
            for move in self.move_candidates():
                elapsed_seconds = (datetime.now() - start_time).total_seconds()
                # Check if there is still time to run algorithm
                if elapsed_seconds > self.options.max_time:
                    print("AI ran out of time")
                    logging.info(f'AI ran out of time')
                    break
                self_clone = self.clone()
                self_clone.perform_move(move, True)
                eval = self_clone.minimax_withab(depth - 1, True, alpha, beta)
                if eval[0] < min_eval:
                    min_eval = eval[0]
                    best_move = move
                beta = min(beta, eval[0])
                if beta <= alpha:
                    break # Alpha cutoff
                count += 1
            return min_eval, best_move, count

    def minimax(self, depth, max_player):
        """ Function for determining best move for AI to make using minimax algoritm """
        start_time = datetime.now()
        if depth == 0 or self.has_winner() is not None:
            if self.options.heuristic == 0:
                return self.heuristic_e0(), None, depth
            if self.options.heuristic == 1:
                return self.heuristic_e1(), None, depth
            if self.options.heuristic == 2:
                return self.heuristic_e2(), None, depth

        if max_player:
            max_eval = -float('inf')
            best_move = None
            count = 0
            for move in self.move_candidates():
                elapsed_seconds = (datetime.now() - start_time).total_seconds()
                # Check if there is still time to run algorithm
                if elapsed_seconds > self.options.max_time:
                    print("AI ran out of time")
                    logging.info(f'AI ran out of time')
                    break
                self_clone = self.clone()
                self_clone.perform_move(move, True)
                eval = self_clone.minimax(depth - 1, False)
                if eval[0] > max_eval:
                    max_eval = eval[0]
                    best_move = move
                count += 1
            return max_eval, best_move, count
        else:
            min_eval = float('inf')
            best_move = None
            count = 0
            for move in self.move_candidates():
                elapsed_seconds = (datetime.now() - start_time).total_seconds()
                # Check if there is still time to run algorithm
                if elapsed_seconds > self.options.max_time:
                    print("AI ran out of time")
                    logging.info(f'AI ran out of time')
                    break
                self_clone = self.clone()
                self_clone.perform_move(move, True)
                eval = self_clone.minimax(depth - 1, True)
                if eval[0] < min_eval:
                    min_eval = eval[0]
                    best_move = move
                count += 1
            return min_eval, best_move, count



    def next_turn(self):
        """Transitions game to the next turn."""
        self.next_player = self.next_player.next()
        self.turns_played += 1

    def to_string(self) -> str:
        """Pretty text representation of the game."""
        dim = self.options.dim
        output = ""
        output += f"Next player: {self.next_player.name}\n"
        output += f"Turns played: {self.turns_played}\n"
        coord = Coord()
        output += "\n   "
        for col in range(dim):
            coord.col = col
            label = coord.col_string()
            output += f"{label:^3} "
        output += "\n"
        for row in range(dim):
            coord.row = row
            label = coord.row_string()
            output += f"{label}: "
            for col in range(dim):
                coord.col = col
                unit = self.get(coord)
                if unit is None:
                    output += " .  "
                else:
                    output += f"{str(unit):^3} "
            output += "\n"
        return output

    def __str__(self) -> str:
        """Default string representation of a game."""
        return self.to_string()
    
    def is_valid_coord(self, coord: Coord) -> bool:
        """Check if a Coord is valid within out board dimensions."""
        dim = self.options.dim
        if coord.row < 0 or coord.row >= dim or coord.col < 0 or coord.col >= dim:
            return False
        return True

    def read_move(self) -> CoordPair:
        """Read a move from keyboard and return as a CoordPair."""
        while True:
            s = input(F'Player {self.next_player.name}, enter your move: ')
            coords = CoordPair.from_string(s)
            if coords is not None and self.is_valid_coord(coords.src) and self.is_valid_coord(coords.dst):
                return coords
            else:
                print('Invalid coordinates! Try again.')

    def human_turn(self):
        """Human player plays a move (or get via broker)."""
        if self.options.broker is not None:
            print("Getting next move with auto-retry from game broker...")
            while True:
                mv = self.get_move_from_broker()
                if mv is not None:
                    (success,result) = self.perform_move(mv, False)
                    print(f"Broker {self.next_player.name}: ",end='')
                    print(result)
                    if success:
                        self.next_turn()
                        break
                sleep(0.1)
        else:
            while True:
                mv = self.read_move()
                (success,result) = self.perform_move(mv, False)
                if success:
                    print(f"Player {self.next_player.name}: ",end='')
                    print(result)
                    self.next_turn()
                    break
                else:
                    print("The move is not valid! Try again.")

    def computer_turn(self) -> CoordPair | None:
        """Computer plays a move."""
        mv = self.suggest_move()
        if mv is not None:
            (success,result) = self.perform_move(mv, False)
            if success:
                print(f"Computer {self.next_player.name}: ",end='')
                print(result)
                self.next_turn()
        return mv

    def player_units(self, player: Player) -> Iterable[Tuple[Coord,Unit]]:
        """Iterates over all units belonging to a player."""
        for coord in CoordPair.from_dim(self.options.dim).iter_rectangle():
            unit = self.get(coord)
            if unit is not None and unit.player == player:
                yield (coord,unit)

    def is_finished(self) -> bool:
        """Check if the game is over."""
        return self.has_winner() is not None

    def has_winner(self) -> Player | None:
        """Check if the game is over and returns winner"""
        if self.options.max_turns is not None and self.turns_played >= self.options.max_turns:
            return Player.Defender
        if self._attacker_has_ai:
            if self._defender_has_ai:
                return None
            else:
                return Player.Attacker    
        return Player.Defender

    def move_candidates(self) -> Iterable[CoordPair]:
        """Generate valid move candidates for the next player."""
        move = CoordPair()
        for (src,_) in self.player_units(self.next_player):
            move.src = src
            for dst in src.iter_adjacent():
                move.dst = dst
                if self.is_valid_move(move):
                    yield move.clone()
            move.dst = src
            yield move.clone()

    def random_move(self) -> Tuple[int, CoordPair | None, float]:
        """Returns a random move."""
        move_candidates = list(self.move_candidates())
        random.shuffle(move_candidates)
        if len(move_candidates) > 0:
            return (0, move_candidates[0], 1)
        else:
            return (0, None, 0)

    def suggest_move(self) -> CoordPair | None:
        """Suggest the next move using minimax alpha beta"""
        start_time = datetime.now()
        if self.options.alpha_beta:
            if self.next_player is Player.Attacker:
                (score, move, evals) = self.minimax_withab(depth = self.options.max_depth, max_player=True, alpha=-float('inf'), beta=float('inf'))
            else:
                (score, move, evals) = self.minimax_withab(depth = self.options.max_depth, max_player=False, alpha=-float('inf'), beta=float('inf'))
        else:
            if self.next_player is Player.Attacker:
                (score, move, evals) = self.minimax(depth = self.options.max_depth, max_player=True)
            else:
                (score, move, evals) = self.minimax(depth = self.options.max_depth, max_player=False)
        elapsed_seconds = (datetime.now() - start_time).total_seconds()
        self.stats.total_seconds += elapsed_seconds
        print(f"Heuristic score: {score}")
        logging.info(f'Heuristic score: {score}')
        logging.info (f'Time for this action: {elapsed_seconds}')
        for k in sorted(self.stats.evaluations_per_depth.keys()):
            print(f"{k}:{self.stats.evaluations_per_depth[k]} ",end='')
        print()
        self.total_evals += evals
        logging.info(f'Cumulative evals: {self.total_evals}')
        #for i in range(self.evals_by_depth):
            #logging.info(f'Cumulative evals by depth: {i}={self.evals_by_depth[i]}')
        if self.stats.total_seconds > 0:
            print(f"Eval perf.: {self.total_evals/self.stats.total_seconds/1000:0.1f}k/s")
        print(f"Elapsed time: {elapsed_seconds:0.1f}s")
        return move

    def post_move_to_broker(self, move: CoordPair):
        """Send a move to the game broker."""
        if self.options.broker is None:
            return
        data = {
            "from": {"row": move.src.row, "col": move.src.col},
            "to": {"row": move.dst.row, "col": move.dst.col},
            "turn": self.turns_played
        }
        try:
            r = requests.post(self.options.broker, json=data)
            if r.status_code == 200 and r.json()['success'] and r.json()['data'] == data:
                # print(f"Sent move to broker: {move}")
                pass
            else:
                print(f"Broker error: status code: {r.status_code}, response: {r.json()}")
        except Exception as error:
            print(f"Broker error: {error}")

    def get_move_from_broker(self) -> CoordPair | None:
        """Get a move from the game broker."""
        if self.options.broker is None:
            return None
        headers = {'Accept': 'application/json'}
        try:
            r = requests.get(self.options.broker, headers=headers)
            if r.status_code == 200 and r.json()['success']:
                data = r.json()['data']
                if data is not None:
                    if data['turn'] == self.turns_played+1:
                        move = CoordPair(
                            Coord(data['from']['row'],data['from']['col']),
                            Coord(data['to']['row'],data['to']['col'])
                        )
                        print(f"Got move from broker: {move}")
                        return move
                    else:
                        # print("Got broker data for wrong turn.")
                        # print(f"Wanted {self.turns_played+1}, got {data['turn']}")
                        pass
                else:
                    # print("Got no data from broker")
                    pass
            else:
                print(f"Broker error: status code: {r.status_code}, response: {r.json()}")
        except Exception as error:
            print(f"Broker error: {error}")
        return None

##############################################################################################################

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(
        prog='ai_wargame',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--max_depth', type=int, help='maximum search depth')
    parser.add_argument('--max_time', type=float, help='maximum search time')
    parser.add_argument('--game_type', type=str, default = 'auto',help='game type: auto|attacker|defender|manual')
    parser.add_argument('--broker', type=str, help='play via a game broker')
    args = parser.parse_args()

    # parse the game type
    if args.game_type == "attacker":
        game_type = GameType.AttackerVsComp
    elif args.game_type == "defender":
        game_type = GameType.CompVsDefender
    elif args.game_type == "manual":
        game_type = GameType.AttackerVsDefender
    else:
        game_type = GameType.CompVsComp

    # set up game options
    options = Options(game_type=game_type)

    # override class defaults via command line options
    if args.max_depth is not None:
        options.max_depth = args.max_depth
    if args.max_time is not None:
        options.max_time = args.max_time
    if args.broker is not None:
        options.broker = args.broker

    # create a new game
    game = Game(options=options)

    # User input of the maximum number of turns
    # Prompt the user to input a value until a valid integer is written 
    while True:
        try:
            max_turns = int(input("Enter the maximum number of turns: "))
            options.max_turns = max_turns
            break
        # Output error if the input value is not an integer
        except ValueError:
            print("Invalid! Please enter a valid integer.")

    # User input of the maximum allowed time for the AI to return a move
    # Prompt the user to input a value until a valid float is written 
    while True:
        try:
            max_time = float(input("Enter the maximum allowed time for the AI to return a move: "))
            options.max_time = max_time
            break
        # Output error if the input value is not a float
        except ValueError:
            print("Invalid! Please enter a valid float.")

    # User input (TRUE or FALSE) for use of either minimax (FALSE) or alpha-beta (TRUE)
    # Prompt the user to input either true or false 
    while True:
        alpha_beta = (input("Enter false (minimax) or true (alpha-beta): "))
        if alpha_beta in ["true", "false"]:
            options.alpha_beta = alpha_beta == "true"
            break
        # Output error if the input value is not from the allowed inputs
        else:
            print("Invalid! Please enter either true or false.")

    # User input for play mode
    # Prompt the user to input either attacker (H-AI), defender (AI-H), manual (H-H), or ai (AI-AI)
    while True:
        game_type = (input("Enter enter the play mode (attacker, defender, manual, or ai): "))
        if game_type == "attacker":
            options.game_type = GameType.AttackerVsComp
            break
        elif game_type == "defender":
            options.game_type = GameType.CompVsDefender
            break
        elif game_type == "manual":
            options.game_type = GameType.AttackerVsDefender
            break
        elif game_type == "ai":
            options.game_type = GameType.CompVsComp
            break
        # Output error if the input value is not from the allowed inputs
        else:
            print("Invalid! Please enter a valid input.")
    
    # Select which heuristic to use
    if options.game_type is not GameType.AttackerVsDefender:
        while True:
            heuristic = (input("Enter which heuristic to use (e0, e1 or e2): "))
            if heuristic == "e0":
                options.heuristic = 0
                break
            elif heuristic == "e1":
                options.heuristic = 1
                break
            elif heuristic == "e2":
                options.heuristic = 2
                break
            else:
                print("Invalid! Please enter a valid input.")

        while True:
            depth = int(input("Enter the depth for the minimax algorithms: "))
            options.max_depth = depth
            break

    logging.basicConfig(filename=f'gameTrace-{options.alpha_beta}-{options.max_time}-{options.max_turns}.txt', filemode='w', level=logging.INFO)

    logging.info(f'---GAME PARAMETERS---\n')
    logging.info(f'Timeout Value: {options.max_time} s\n')
    logging.info(f'Maximum number of turns: {options.max_turns}\n')
    logging.info(f'Alpha-beta status: {options.alpha_beta}\n')

    if options.game_type == GameType.AttackerVsComp:
        logging.info(f'Play Mode: player 1 = H & player 2 = AI\n')
    if options.game_type == GameType.CompVsDefender:
        logging.info(f'Play Mode: player 1 = AI & player 2 = H\n')
    if options.game_type == GameType.AttackerVsDefender:
        logging.info(f'Play Mode: player 1 = H & player 2 = H\n')
    if options.game_type == GameType.CompVsComp:
        logging.info(f'Play Mode: player 1 = AI & player 2 = AI\n')
    # log the heuristic used if a player is an AI
    if options.game_type != GameType.AttackerVsDefender and options.heuristic is not None:
        logging.info(f'Heuristic used: {options.heuristic}')
    else:
        logging.info(f'No heuristics used')
    logging.info("\n")
    logging.info(f'---Initial Configuration of the Game---\n')
    logging.info("\n")

    # the main game loop
    while True:
        print()
        #print the board
        print(game.to_string())
        logging.info(game.to_string())
        winner = game.has_winner()
        if winner is not None:
            print(f"{winner.name} wins!")
            logging.info(f'{winner.name} wins in {game.turns_played} turns\n')

            break
        if game.options.game_type == GameType.AttackerVsDefender:
            game.human_turn()
        elif game.options.game_type == GameType.AttackerVsComp and game.next_player == Player.Attacker:
            game.human_turn()
        elif game.options.game_type == GameType.CompVsDefender and game.next_player == Player.Defender:
            game.human_turn()
        else:
            player = game.next_player
            move = game.computer_turn()
            if move is not None:
                game.post_move_to_broker(move)
            else:
                print("Computer doesn't know what to do!!!")
                exit(1)

##############################################################################################################

if __name__ == '__main__':
    main()
