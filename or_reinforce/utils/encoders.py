"""Handles encoding part of the state"""
import torch
import typing
import numpy as np

from optimax_rogue.game.state import GameState
from optimax_rogue.logic.moves import Move
from optimax_rogue.game.world import Tile, Dungeon, World

class FlatEncoder:
    """Interface for flat encoders. Assumes that this makes a flat encoding, meaning
    that the game state + move correspond to a single vector"""

    @property
    def dim(self) -> int:
        """The dimensionality of the encoded dimension"""
        raise NotImplementedError

    def encode(self, game_state: GameState, move: Move, out: torch.tensor = None) -> torch.tensor:
        """Encode the game state and move into a tensor"""
        raise NotImplementedError

class MergedFlatEncoders(FlatEncoder):
    """Merges multiple encoders together by mapping each encoder to a particular
    section of the tensor, in order

    Attributes:
        children (list[FlatEncoder]): the encoders that this is merging

        _dim (int): cached sum of dim of children
    """

    def __init__(self, children: typing.List[FlatEncoder]) -> None:
        self.children = children
        self._dim = sum((child.dim for child in self.children), 0)

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, game_state: GameState, move: Move, out: torch.tensor = None) -> torch.tensor:
        if out is None:
            out = torch.zeros((self._dim,), dtype=torch.float)

        ind = 0
        for child in self.children:
            child.encode(game_state, move, out[ind:ind + child.dim])
            ind += child.dim
        return out

class MoveEncoder(FlatEncoder):
    """Maps moves 1-hot to the given vector.

    Attributes:
        moves (list[Move]): the moves that we support

        _lookup (dict[Move, int]): maps moves to ints
    """
    def __init__(self, moves: typing.List[Move]) -> None:
        self.moves = moves
        self._lookup = dict((move, ind) for ind, move in enumerate(moves))

    @property
    def dim(self) -> int:
        return len(self.moves)

    def encode(self, game_state: GameState, move: Move, out: torch.tensor = None) -> torch.tensor:
        if out is None:
            out = torch.zeros((self.dim,), dtype=torch.float)

        out[:] = 0
        out[self._lookup[move]] = 1
        return out

class SurroundBarrierEncoder(FlatEncoder):
    """Maps the surrounding tiles into the tensor, row-by-row, 1-hot by tile
    type.

    Attributes:
        entity_iden (int): the entity we are looking from
        view_distance (int): how many tiles out to see. for a view distance
            of 1, you get the surrounding 8 tiles

        _lookup (dict[Tile, int]): maps tiles to the lookup for 1-hot encoding
    """

    def __init__(self, entity_iden: int, view_distance: int):
        self.entity_iden = entity_iden
        self.view_distance = view_distance
        self._lookup = dict((tile, ind) for ind, tile in enumerate(Tile))

        self._dim = ((view_distance * 2 + 1) ** 2) * len(self._lookup)

    @property
    def dim(self):
        return self._dim

    def encode(self, game_state: GameState, move: Move, out: torch.tensor = None) -> torch.tensor:
        if out is None:
            out = torch.zeros((self.dim,), dtype=torch.float)
        else:
            out[:] = 0

        if self.entity_iden not in game_state.iden_lookup:
            return out

        ent = game_state.iden_lookup[self.entity_iden]
        if ent.depth not in game_state.world.dungeons:
            return out
        dung: Dungeon = game_state.world.dungeons[ent.depth]

        out = out.view(self.view_distance * 2 + 1,
                       self.view_distance * 2 + 1,
                       len(self._lookup))
        for x_ind in range(self.view_distance * 2 + 1):
            x = ent.x + (x_ind - self.view_distance)
            if x < 0 or x >= dung.width:
                continue
            for y_ind in range(self.view_distance * 2 + 1):
                y = ent.y + (y_ind - self.view_distance)
                if y < 0 or y >= dung.height:
                    continue
                out[x_ind, y_ind, self._lookup[int(dung.tiles[x, y])]] = 1

        return out.view(self.dim)

class LocationEncoder(FlatEncoder):
    """Encodes the location of the specified entity as an (x, y) pair

    Attributes:
        entity_iden (int): the iden for the entity
    """
    def __init__(self, entity_iden: int):
        self.entity_iden = entity_iden

    @property
    def dim(self):
        return 2

    def encode(self, game_state: GameState, move: Move, out: torch.tensor = None) -> torch.tensor:
        if out is None:
            out = torch.zeros((self.dim,), dtype=torch.float)

        ent = game_state.iden_lookup[self.entity_iden]
        out[0] = ent.x
        out[1] = ent.y
        return out

class StaircaseLocationEncoder(FlatEncoder):
    """Encodes the location of the staircase on the same level as the entity as
    an (x, y) pair

    Attributes:
        entity_iden (int): the entity whose depth is used to determine the location
            of the staircase
    """
    def __init__(self, entity_iden: int):
        self.entity_iden = entity_iden

    @property
    def dim(self):
        return 2

    def encode(self, game_state: GameState, move: Move, out: torch.tensor = None) -> torch.tensor:
        if out is None:
            out = torch.zeros((self.dim,), dtype=torch.float)

        ent = game_state.iden_lookup[self.entity_iden]
        dung = game_state.world.get_at_depth(ent.depth)
        x, y = dung.staircase()
        out[0] = x
        out[1] = y
        return out


class StaircaseEncoder(FlatEncoder):
    """Encodes an 'arrow' to the staircase via an angle to the staircase from
    a given entity

    Attributes:
        entity_iden (int): the iden for the entity the arrow points from
    """
    def __init__(self, entity_iden: int):
        self.entity_iden = entity_iden

    @property
    def dim(self):
        return 1

    def encode(self, game_state: GameState, move: Move, out: torch.tensor = None) -> torch.tensor:
        if out is None:
            out = torch.zeros((self.dim,), dtype=torch.float)

        if self.entity_iden not in game_state.iden_lookup:
            return out

        ent = game_state.iden_lookup[self.entity_iden]

        if ent.depth not in game_state.world.dungeons:
            return out

        dung: Dungeon = game_state.world.get_at_depth(ent.depth)

        scx, scy = dung.staircase()

        delx, dely = scx - ent.x, scy - ent.y
        out[0] = float(np.arctan2(dely, delx))
        return out
