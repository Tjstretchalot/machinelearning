"""Some tests for encoders can go here"""
import torch

import optimax_rogue.game.state as state
import optimax_rogue.game.world as world
import optimax_rogue.logic.worldgen as worldgen
import optimax_rogue.game.entities as entities
import optimax_rogue.logic.moves as moves
import or_reinforce.utils.encoders as encs

def make_state() -> state.GameState:
    """Creates a semi-random state"""
    dung: world.Dungeon = worldgen.EmptyDungeonGenerator(20, 20).spawn_dungeon(0)
    p1x, p1y = dung.get_random_unblocked()
    p2x, p2y = dung.get_random_unblocked()
    while (p2x, p2y) == (p1x, p1y):
        p2x, p2y = dung.get_random_unblocked()
    ent1 = entities.Entity(1, 0, p1x, p1y, 10, 10, 2, 1, [], dict())
    ent2 = entities.Entity(2, 0, p2x, p2y, 10, 10, 2, 1, [], dict())
    return state.GameState(True, 1, 1, 2, world.World({0: dung}), [ent1, ent2])


def _staircase_onehot_test1():
    dung: world.Dungeon = worldgen.EmptyDungeonGenerator(20, 20).spawn_dungeon(0)
    while dung.staircase()[0] <= 1:
        dung: world.Dungeon = worldgen.EmptyDungeonGenerator(20, 20).spawn_dungeon(0)

    p1x, p1y = dung.staircase()
    p1x -= 1

    p2x, p2y = dung.get_random_unblocked()
    while (p2x, p2y) == (p1x, p1y):
        p2x, p2y = dung.get_random_unblocked()
    ent1 = entities.Entity(1, 0, p1x, p1y, 10, 10, 2, 1, [], dict())
    ent2 = entities.Entity(2, 0, p2x, p2y, 10, 10, 2, 1, [], dict())
    gstate = state.GameState(True, 1, 1, 2, world.World({0: dung}), [ent1, ent2])

    onehot = encs.StaircaseDirectionOneHotEncoder(1, 1)
    truth = torch.tensor([0, 0, 1, 0, 1, 0], dtype=torch.float)
    got = onehot.encode(gstate, None)
    if (torch.isclose(got, truth) == 0).sum() != 0:
        raise ValueError(f'got {got}, truth {truth}')

def _staircase_onehot_test2():
    dung: world.Dungeon = worldgen.EmptyDungeonGenerator(20, 20).spawn_dungeon(0)
    while dung.staircase()[0] <= 1:
        dung: world.Dungeon = worldgen.EmptyDungeonGenerator(20, 20).spawn_dungeon(0)

    p1x, p1y = dung.staircase()
    p1x -= 1

    p2x, p2y = dung.get_random_unblocked()
    while (p2x, p2y) == (p1x, p1y):
        p2x, p2y = dung.get_random_unblocked()
    ent1 = entities.Entity(1, 0, p1x, p1y, 10, 10, 2, 1, [], dict())
    ent2 = entities.Entity(2, 0, p2x, p2y, 10, 10, 2, 1, [], dict())
    gstate = state.GameState(True, 1, 1, 2, world.World({0: dung}), [ent1, ent2])

    onehot = encs.StaircaseDirectionOneHotEncoder(2, 1)
    truth = torch.tensor([0, 0, 0, 1, 0, 0, 0, 1, 0, 0], dtype=torch.float)
    got = onehot.encode(gstate, None)
    if (torch.isclose(got, truth) == 0).sum() != 0:
        raise ValueError(f'got {got}, truth {truth}')

if __name__ == '__main__':
    _staircase_onehot_test1()
    _staircase_onehot_test2()
