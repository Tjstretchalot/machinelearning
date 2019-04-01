"""The shared sensitives within this package. By sharing sensitives it is possible to include
any of this packages approaches or all of them in your evolution algorithm."""

import shared.evolution as evo

def generate() -> evo.EvolutionProblem:
    """Generates the sensitive parameters for tuning"""
    return evo.EvolutionProblem(
        categoricals=[],
        integrals=[evo.NumericEvolvableParam('batch_size', True, (5, 64))],
        continuous=[evo.NumericEvolvableParam('learning_rate', False, (1e-06, 0.1))])
