"""Simple tests for the replay buffer"""
import os
import random
import shared.filetools as filetools
import or_reinforce.deep.replay_buffer as rb
import optimax_rogue.game.state as state
import optimax_rogue.game.world as world
import optimax_rogue.logic.worldgen as worldgen
import optimax_rogue.game.entities as entities
import optimax_rogue.logic.moves as moves


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

def make_exp() -> rb.Experience:
    """Creates a semi-random experience"""
    lmoves = list(moves.Move)
    return rb.Experience(make_state(), random.choice(lmoves), random.random(), 1)

FILEPATH = os.path.join('out', 'or_reinforce', 'deep', 'tests', 'replay_buffer_test')
def main():
    """Main entry for tests"""

    if os.path.exists(FILEPATH):
        filetools.deldir(FILEPATH)

    buf = rb.FileWritableReplayBuffer(os.path.join(FILEPATH, '1'), exist_ok=False)

    sbuf = []

    for _ in range(5):
        exp = make_exp()
        buf.add(exp)
        sbuf.append(exp)

    buf2 = rb.FileWritableReplayBuffer(os.path.join(FILEPATH, '2'), exist_ok=False)

    for _ in range(5):
        exp = make_exp()
        buf2.add(exp)
        sbuf.append(exp)

    buf.close()
    buf2.close()

    rb.merge_buffers([os.path.join(FILEPATH, '2'), os.path.join(FILEPATH, '1')], os.path.join(FILEPATH, '3'))

    buf.close()
    buf = rb.FileReadableReplayBuffer(os.path.join(FILEPATH, '3'))

    for _ in range(3):
        missing = [exp for exp in sbuf]
        for _ in range(10):
            got = buf.sample(1)[0]
            for i in range(len(missing)): #pylint: disable=consider-using-enumerate
                if got == missing[i]:
                    missing.pop(i)
                    break
            else:
                raise ValueError(f'got bad value: {got} expected one of \n'
                        + '\n'.join(repr(exp) for exp in missing))

    buf.mark()
    got = buf.sample(1)[0]
    buf.reset()
    got2 = buf.sample(1)[0]
    if got != got2:
        raise ValueError(f'mark did not retrieve same experience: {got} vs {got2}')

    buf.close()

if __name__ == '__main__':
    main()