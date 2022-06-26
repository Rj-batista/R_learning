import math
import random
import itertools

import numpy as np
import numba
import numpy.random
from more_itertools import take
from tqdm import tqdm


class SingleAgentEnv:
    def state_id(self) -> int:
        pass

    def is_game_over(self) -> bool:
        pass

    def act_with_action_id(self, action_id: int):
        pass

    def score(self) -> float:
        pass

    def available_actions_ids(self) -> np.ndarray:
        pass

    def reset(self):
        pass

    def view(self):
        pass

    def reset_random(self):
        pass


class LineWorld(SingleAgentEnv):
    def __init__(self, nb_cells: int = 5):
        self.nb_cells = nb_cells
        self.current_cell = math.floor(nb_cells / 2)
        self.step_count = 0

    def state_id(self) -> int:
        return self.current_cell

    def is_game_over(self) -> bool:
        if self.step_count > self.nb_cells * 2:
            return True
        return self.current_cell == 0 or self.current_cell == self.nb_cells - 1

    def act_with_action_id(self, action_id: int):
        self.step_count += 1
        if action_id == 0:
            self.current_cell -= 1
        else:
            self.current_cell += 1

    def score(self) -> float:
        if self.current_cell == 0:
            return -1.0
        elif self.current_cell == self.nb_cells - 1:
            return 1.0
        else:
            return 0.0

    def available_actions_ids(self) -> np.ndarray:
        return np.array([0, 1])

    def reset(self):
        self.current_cell = math.floor(self.nb_cells / 2)
        self.step_count = 0

    def view(self):
        print(f'Game Over: {self.is_game_over()}')
        print(f'score : {self.score()}')
        for i in range(self.nb_cells):
            if i == self.current_cell:
                print("X", end='')
            else:
                print("_", end='')
        print()

    def reset_random(self):
        self.current_cell = random.randint(0, self.nb_cells - 1)
        self.step_count = 0


def q_learning(env: SingleAgentEnv, max_iter_count: int = 10000,
               gamma: float = 0.99,
               alpha: float = 0.1,
               epsilon: float = 0.2):
    q = {}

    for it in range(max_iter_count):

        if env.is_game_over():
            env.reset()

        s = env.state_id()
        aa = env.available_actions_ids()

        if s not in q:
            q[s] = {}
            for a in aa:
                q[s][a] = 0.0 if env.is_game_over() else random.random()

        if random.random() <= epsilon:
            a = np.random.choice(aa)
        else:
            a = aa[np.argmax([q[s][a] for a in aa])]

        old_score = env.score()
        env.act_with_action_id(a)
        new_score = env.score()
        r = new_score - old_score

        s_p = env.state_id()
        aa_p = env.available_actions_ids()

        if s_p not in q:
            q[s_p] = {}
            for a in aa_p:
                q[s_p][a] = 0.0 if env.is_game_over() else random.random()

        q[s][a] += alpha * (r + gamma * np.max([q[s_p][a] for a in aa_p]) - q[s][a])

    pi = {}
    for (s, a_dict) in q.items():
        pi[s] = {}
        actions = []
        q_values = []
        for (a, q_value) in a_dict.items():
            actions.append(a)
            q_values.append(q_value)

        best_action_idx = np.argmax(q_values)
        for i in range(len(actions)):
            pi[s][actions[i]] = 1.0 if actions[i] == best_action_idx else 0.0

    return q, pi


print(q_learning(LineWorld(5), epsilon=1.0, max_iter_count=10000))

