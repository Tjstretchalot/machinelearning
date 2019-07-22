# Optimax Rogue Client

This section is meant for reinforcement learning in OptiMAX Rogue. The server,
some dummy clients, and a command-line spectator are available in the following
repository: https://github.com/tjstretchalot/optimax_rogue

A prettier client which bots can be connected to is available at
https://github.com/tjstretchalot/ORClient

## Summary of Bots

The latest model is deep/deep2.py. The order of development was simple/simplebot.py,
simple/simplebot2.py, simple/simplebot3.py, deep/deep1.py, then deep/deep2.py.

The models got more generic as numbers decrease, with simplebot and deep1 having most
of their properties hardcoded whereas simplebot3 and deep2 are somewhat more generic.

deep2 was the first bot to *not* encode the move as an input, but instead have the
one output feature per possible move. This lends towards much simpler analysis and
smoother dynamics. Correspondingly, deep2 is able to handle a much more intricate
input space.

## Summary of Features

- Replay Balancing: deep/replay_buffer.py
- Replay Buffer: deep/replay_buffer.py
- Prioritized Replay: deep/replay_buffer.py, deep/deep2.py
- Target Network: Inside each bot file
- Annealed Parameters / Epsilon Greedy: qbot.py, runners/deep_trainer.py
- Shaped Reward: rewarders.py
- Relative Game State Encoding: encoders.py
- Analysis: runners/deep2_analyze.py

## Usage

The following .sh file can be used to train the deep2 bot:

```
nice python3 -u -m or_reinforce.runners.deep_trainer --py3 --numthreads 40 --headless or_reinforce.deep.deep2.deep2 2>&1 | tee log.txt
```

Going through this in order

- `nice` - see http://manpages.ubuntu.com/manpages/trusty/man1/nice.1.html
- `python3` - the python v3.6+ executable
- `-u` - used to see output quickly. Required because python aggressively
buffers output when piped to other processes by default.
- `-m` - tells python to invoke the passed module while remaining relative
to the current module
- `or_reinforce.runners.deep_trainer` - the module that trains deep-style bots
- `--py3` - tells the deep_trainer to spawn new processes with `python3` instead
of `python`
- `--numthreads 40` - how many threads to use for gathering experiences. Should be
1-5x the number of cores in my experience. Keep increasing this until you see about
75% cpu usage when gathering experiences.
- `--headless` - If not specified you should have a very low (i.e., 1) number of threads.
lets you view the experiences as they are gathered using the default command line spectator.
- `or_reinforce.deep.deep2.deep2` - the bot module to train (`or_reinforce.deep.deep2`) with
the callable inside the module that returns an instance of a Bot (`.deep2`)
- `2>&1` - Redirect std error to std output so we can redirect both together
- `| tee log.txt` - Take the standard output (merged with std err above) and store it in
log.txt as well as print it out to standard output. This lets you view the result with
screen or running it directly, or by tailing log.txt. Additionally, after-the-fact, you
can view the logs by looking at log.txt

Once training completes, the final model that is ready for evaluation will be stored at
`out/or_reinforce/deep/deep2/model_eval`. It can be loaded as follows:

```py
import or_reinforce.deep.deep2 as deep2
mod = deep2.init_or_load_model(evaluation=True)
```

And it can be evaluated like any torch model:

```py
import torch
test_inps = torch.randn((1, deep2.ENCODE_DIM))
outs = mod(test_inps)
```

Of course with random inputs the output will be nonsense, but that gets the idea across.
For more sophisticated analysis, I use the following command:

```
nice python3 -u -m or_reinforce.runners.deep2_analyze --py3 --numthreads 40 --numexps 4000 --train_force_amount 0.1 --pca3d 2>&1 | tee log.txt
```

This is much the same as the training command. `--train_force_amount` is the percent of moves
which are chosen randomly when gathering experiences for analysis. It should be large enough
that a good sample of possible states are found. Note that `--pca3d` can take 10 minutes to
calculate, and corresponds to a high-def video of the samples projected into the top-3 principal
component space for the network. See `shared/measures/pca_3d.py` for details.