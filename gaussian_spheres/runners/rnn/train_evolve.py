"""Uses the evolution trainer on the gaussian spheres task"""

import shared.setup_torch #pylint: disable=unused-import
import shared.evolution as evo
import shared.evolvable_models.rnn as ernn
import shared.evolvable_models.sensitives as esens
from shared.models.rnn import NaturalRNN
import shared.filetools
import torch
from gaussian_spheres.pwl import GaussianSpheresPWLP
import os

INPUT_DIM = 200
OUTPUT_DIM = 2
MODULE_NAME = 'gaussian_spheres.runners.train_evolve'
SAVE_PATH = shared.filetools.savepath()

def create_data() -> GaussianSpheresPWLP:
    """Create the task to solve"""
    result = GaussianSpheresPWLP.create(
        epoch_size=1800, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, cube_half_side_len=2,
        num_clusters=60, std_dev=0.04, mean=0, min_sep=0.1
    )
    return result, result

def generate() -> evo.CompleteEvolutionProblem:
    """Generates the complete problem - approaches and sensitives"""
    return evo.CompleteEvolutionProblem(
        INPUT_DIM, OUTPUT_DIM,
        evo.CategoricalEvolvableParam(
            'rnn', [ernn.RNNApproach(create_data)]),
        esens.generate()
    )

def main():
    """Main entry"""
    evo.Evolver(evo.EvolverSettings(
        problem_import=MODULE_NAME,
        problem_func='generate',
        save_path=SAVE_PATH,
        duration_ms=1,
        cores=6,
        metric_name='accuracy',
        max_trials=100,
        trial_patience=1,
        trial_epsilon=1e-06,
        seed_points=5,
        salient_points=5,
        evolve_patience=5,
        evolve_epsilon=1e-06,
        pass_patience=1,
        max_passes=10
    )).run()

if __name__ == '__main__':
    main()