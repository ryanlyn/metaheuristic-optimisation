import os
import sys
import time

import numpy as np 

class ParticleSwarm:
    _version = '1.6'
    ω = 1
    φ_p = 2
    φ_g = 2

    def __init__(self, fitness_func, dimensions, X_range, pop_size=100, random_state=None, vectorised=False):
        self.fitness_func = fitness_func
        self.dimensions = dimensions

        assert len(X_range) == self.dimensions 
        self.X_min = np.array([x_range[0] for x_range in X_range])
        self.X_max = np.array([x_range[1] for x_range in X_range])
        self.V_min = -np.abs(self.X_max - self.X_min)
        self.V_max = np.abs(self.X_max - self.X_min)

        self.pop_size = pop_size
        self.seed = random_state
        self.vectorised = vectorised

    def __repr__(self):
        string = f"{self.__class__.__name__}(fitness_func, dimensions={self.dimensions}, X_range, pop_size={self.pop_size}, random_state={self.seed}, vectorised={self.vectorised})"
        return string

    def __str__(self):
        string = f"{self.__class__.__name__}(fitness_func, dimensions={self.dimensions}, X_range, pop_size={self.pop_size}, random_state={self.seed}, vectorised={self.vectorised})"
        return string

    def _initialise(self):
        np.random.seed(self.seed)

        positions = np.random.uniform(low=self.X_min, high=self.X_max, size=(self.pop_size, self.dimensions))

        if self.vectorised is True:
            evals = self.fitness_func(*positions.T)
        else:
            evals = []
            for p in positions:
                evals.append(self.fitness_func(*p))
            evals = np.array(evals)

        best_evals = evals
        best_positions = positions

        swarm_eval = best_evals.min()
        swarm_position = best_positions[best_evals.argmin()]

        assert self.fitness_func(*swarm_position) == swarm_eval

        velocities = np.random.uniform(low=self.V_min, high=self.V_max, size=(self.pop_size, self.dimensions))

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
        prev_best_evals = output_dict['best_evals']
        prev_velocities = output_dict['velocities']
        prev_swarm_eval = output_dict['swarm_eval']
        prev_swarm_position = output_dict['swarm_position']
        
        ρ_p = np.random.uniform(low=0, high=1, size=(self.pop_size, self.dimensions))
        ρ_g = np.random.uniform(low=0, high=1, size=(self.pop_size, self.dimensions))

        velocities = self.ω * prev_velocities + self.φ_p * ρ_p * (prev_best_positions - prev_positions) + self.φ_g * ρ_g * (prev_swarm_position - prev_positions)

        positions = prev_positions + velocities

        if self.vectorised is True:
            evals = self.fitness_func(*positions.T)
        else:
            evals = []
            for p in positions:
                evals.append(self.fitness_func(*p))
            evals = np.array(evals)

        best_evals = np.where((evals < prev_best_evals), evals, prev_best_evals)
        positions_mask = np.repeat((evals < prev_best_evals)[:, np.newaxis], self.dimensions, axis=-1)
        best_positions = np.where(positions_mask, positions, prev_best_positions)

        best_new_eval = best_evals.min()
        best_new_position = best_positions[best_evals.argmin()]

        if best_new_eval < prev_swarm_eval:
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

    def optimise(self, n_iterations=1000, verbose=True):
        start = time.time()
        output_dict = self._initialise()

        iteration = 1
        while iteration <= n_iterations:
            _  = self._update_particle_vector(output_dict)
            output_dict = _

            if verbose is True:
                string_1 = f"Updated iteration {iteration} out of {n_iterations} ({iteration/n_iterations:.1%})"
                string_2 = f"Best fitness score: {output_dict['swarm_eval']}"
                sys.stdout.flush()
                sys.stdout.write(string_1 + ' --- ' + string_2 + '\r')

            iteration += 1

        end = time.time()
        total_time = end - start

        results = {'best_score': output_dict['swarm_eval'],
                   'best_position': output_dict['swarm_position'],
                   'total_time': total_time}
        return results
