import os
import sys

import numpy as np 

class ParticleSwarm:
    _version = '1.0'
    ω = 1
    φ_p = 2
    φ_g = 2

    def __init__(self, fitness_func, X_0, X_range, n_particles=100, n_iterations=1000, random_state=None):
        self.fitness_func = fitness_func
        self.X_0 = X_0
        self.n_x = len(X_0)
        self.X_min = np.array([x_range[0] for x_range in X_range])
        self.X_max = np.array([x_range[1] for x_range in X_range])
        self.V_min = -np.abs(self.X_max - self.X_min)
        self.V_max = np.abs(self.X_max - self.X_min)
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.seed = random_state

    def __repr__(self):
        string = f"{self.__class__.__name__}(fitness_func, X_0, X_range, n_particles={self.n_particles}, n_iterations={self.n_iterations})"
        return string

    def __str__(self):
        string = f"{self.__class__.__name__}(fitness_func, X_0, X_range, n_particles={self.n_particles}, n_iterations={self.n_iterations})"
        return string

    def _initialise(self):
        np.random.seed(self.seed)

        positions = np.random.uniform(low=self.X_min, high=self.X_max, size=(self.n_particles, self.n_x))
        best_positions = positions

        evals = self.fitness_func(*best_positions.T)
        best_evals = evals

        swarm_eval = best_evals.max()
        swarm_position = best_positions[best_evals.argmax()]

        assert self.fitness_func(*swarm_position) == swarm_eval

        velocities = np.random.uniform(low=self.V_min, high=self.V_max, size=(self.n_particles, self.n_x))

        output_dict = {'positions': positions,
                       'best_positions': best_positions,
                       'evals': evals,
                       'best_evals': best_evals,
                       'velocities': velocities,
                       'swarm_eval': swarm_eval,
                       'swarm_position': swarm_position}
        return output_dict

    def _update_particle_vector(self, output_dict):
        prev_positions = output_dict['positions']
        prev_best_positions = output_dict['best_positions']
        prev_evals = output_dict['evals']
        prev_best_evals = output_dict['best_evals']
        prev_velocities = output_dict['velocities']
        prev_swarm_eval = output_dict['swarm_eval']
        prev_swarm_position = output_dict['swarm_position']
        
        ρ_p = np.random.uniform(low=0, high=1, size=(self.n_particles, self.n_x))
        ρ_g = np.random.uniform(low=0, high=1, size=(self.n_particles, self.n_x))

        velocities = self.ω * prev_velocities + self.φ_p * ρ_p * (prev_best_positions - prev_positions) + self.φ_g * ρ_g * (prev_swarm_position - prev_positions)
        positions = prev_positions + velocities

        evals = self.fitness_func(*positions.T)
        best_evals = np.where((evals > prev_best_evals), evals, prev_best_evals)
        positions_mask = np.repeat((evals > prev_best_evals)[:, np.newaxis], self.n_x, axis=-1)
        best_positions = np.where(positions_mask, positions, prev_best_positions)

        best_new_eval = best_evals.max()
        best_new_position = best_positions[best_evals.argmax()]

        if best_new_eval > prev_swarm_eval:
            swarm_eval = best_new_eval
            swarm_position = best_new_position
        else:
            swarm_eval = prev_swarm_eval
            swarm_position = prev_swarm_position

        output_dict = {'positions': positions,
                       'best_positions': best_positions,
                       'evals': evals,
                       'best_evals': best_evals,
                       'velocities': velocities,
                       'swarm_eval': swarm_eval,
                       'swarm_position': swarm_position}
        return output_dict

    def optimise(self):
        pass
