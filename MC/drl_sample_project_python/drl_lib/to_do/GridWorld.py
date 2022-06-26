from MC.drl_sample_project_python.drl_lib.do_not_touch.contracts import SingleAgentEnv
import numpy as np
import random


class GridWorld(SingleAgentEnv):
    def __init__(self):
        random.seed(42)
        self.current_cell = random.randint(0, 24)
        self.cols, self.rows = 5, 5
        self.nb_cells = 25
        self.terminal_values = [4, 24]

    def state_id(self) -> int:
        return self.current_cell

    def is_game_over(self) -> bool:
        if self.current_cell == self.cols - 1 or self.current_cell == self.nb_cells - 1:
            return True

    def act_with_action_id(self, action_id: int):
        haut = [0, 1, 2, 3, 4]
        bas = [20, 21, 22, 23, 24]
        gauche = [0, 5, 10, 15, 20]
        droite = [4, 9, 14, 19, 24]
        if action_id == 1:
            if self.current_cell not in gauche:
                self.current_cell -= 1
        elif action_id == 3:
            if self.current_cell not in droite:
                self.current_cell += 1
        elif action_id == 2:
            if self.current_cell not in haut:
                self.current_cell -= self.rows
        elif action_id == 0:
            if self.current_cell not in bas:
                self.current_cell += self.rows

    def score(self) -> float:
        if self.current_cell == self.terminal_values[0]:
            return -3.0
        elif self.current_cell == self.terminal_values[1]:
            return 1.0
        else:
            return 0.0

    def available_actions_ids(self) -> np.ndarray:
        return np.array(0, 1, 2, 3)

    def reset(self):
        self.current_cell = 0

    def view(self):
        print(f' Le score :{self.score()} and Fin de partie :{self.is_game_over()}')
        for i in range(self.nb_cells):
            if i % 5 == 0:
                print('\n')
            if i == self.current_cell:
                print('X', end=' ')
            else:
                print('-', end=' ')

    def reset_random(self):
        self.current_cell = random.randint(0, self.nb_cells - 1)
