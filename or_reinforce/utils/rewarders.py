"""A QBot for staircasing"""

from optimax_rogue.game.state import GameState
import or_reinforce.utils.qbot as qbot

class SCRewarder(qbot.Rewarder):
    """Rewards solely based on depth"""
    def reward(self, entity_iden: int, oldstate: GameState, newstate: GameState) -> float:
        ent_old = oldstate.iden_lookup[entity_iden]
        ent_new = newstate.iden_lookup[entity_iden]
        if ent_new.depth > ent_old.depth:
            return 1
        if (ent_new.x, ent_new.y) == (ent_old.x, ent_old.y):
            return -1

        staircase_old = oldstate.world.dungeons[ent_old.depth].staircase()
        staircase_new = newstate.world.dungeons[ent_new.depth].staircase()

        manh_old = abs(staircase_old[0] - ent_old.x) + abs(staircase_old[1] - ent_old.y)
        manh_new = abs(staircase_new[0] - ent_new.x) + abs(staircase_new[1] - ent_new.y)

        return 0.5 * (manh_old - manh_new)
