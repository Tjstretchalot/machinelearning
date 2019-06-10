"""A store for the game state at various intervals. Passing this reference
around avoids having to store a million versions of each game state"""

from optimax_rogue.game.state import GameState
import optimax_rogue.networking.serializer as ser

class _CachedGameState:
    """A small wrapper around the game states that are cached for deciding
    when to prune

    Attributes:
        game_state (GameState): the actual cached game state
        references (int): the number of "references" (with explicit reference
            counting) to this cached game state
    """
    def __init__(self, game_state: GameState, references: int):
        self.game_state = game_state
        self.references = references

class GameStateCache:
    """A cache for game states at various times.

    Attributes:
        cache (dict[int, _CachedGameState]) - maps from a tick to
    """

    def __init__(self):
        self.cache = dict()

    def contains(self, tick: int) -> bool:
        """Returns True if this contains the game state cache at the given tick, False
        otherwise"""
        return tick in self.cache

    def put(self, game_state: GameState, references: int = 1):
        """Push the given game state to this cache. This will deep copy the game
        state. Assumes that the caller is holding a reference to this game state,
        but you may specify an alternate number.

        If the game state is already stored, the references are simply incremented

        Args:
            game_state (GameState): the state of the game you want to store
            references (int, optional): Number of references to add. Defaults to 1.
        """
        if self.contains(game_state.tick):
            cachedgstate: _CachedGameState = self.cache[game_state.tick]
            cachedgstate.references += references
            return

        self.cache[game_state.tick] = _CachedGameState(
            ser.deserialize(ser.serialize(game_state)),
            references
        )

    def fetch(self, tick: int) -> GameState:
        """Fetches the game state with the given tick. Raises IndexError if not
        contained in the cache. This does not decrement the references for
        the result (use pop for that)
        """
        return self.cache[tick].game_state

    def pop(self, tick: int, references: int = 1) -> GameState:
        """Fetches the game state with the given tick and decrements its references
        by 1. If no references are there after the call, the game state is freed to
        the GC

        Args:
            tick (int): the tick that you want
            references (optional int): the number of references to remove. Default 1.
        """
        cached: _CachedGameState = self.cache[tick]
        cached.references -= references
        if cached.references <= 0:
            del self.cache[tick]
        return cached.game_state
