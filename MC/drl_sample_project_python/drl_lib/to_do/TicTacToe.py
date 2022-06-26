import random
import numpy as np
from MC.drl_sample_project_python.drl_lib.do_not_touch.contracts import SingleAgentEnv


class TicTacToe(SingleAgentEnv):
    def __init__(self, nb_cells: int = 3**9):
        self.nb_cells = nb_cells
        self.current_cell = 0
        self.step_count = 0
        self.win_rate = 0.0
        self.nterminal = 0

    def state_id(self) -> int:
        return self.current_cell

    def is_game_over(self) -> bool:
        if self.step_count == 9:
            return True
        values = self.values_cases()
        for i in range(3):
            if (values[i] == values[i + 3] and values[i + 3] == values[i + (2 * 3)] and (values[i] == 1 or values[i] == 2)) or \
                    (values[i * 3] == values[(i * 3) + 1] and values[(i * 3) + 1] == values[(i * 3) + 2] and (values[i * 3] == 1 or values[i * 3] == 2)) or \
                    (values[i] == values[4] and values[4] == values[8 - i] and (values[i] == 1 or values[i] == 2)):
                return True
        return False

    def available_actions_ids(self) -> np.ndarray:
        aa = []
        values = self.values_cases()
        for i in range(9):
            if values[i] == 0:
                aa.append(i)
        return np.array(aa)

    def act_with_action_id(self, action_id: int):
        if self.step_count % 2 == 0:
            self.current_cell += 3**(8 - action_id)
        else:
            aa = self.available_actions_ids()
            random_move = random.randint(0, len(aa) - 1)
            self.current_cell += 2 * (3**(8 - aa[random_move]))
        self.step_count += 1

    def score(self) -> float:
        values = self.values_cases()
        for i in range(3):
            if (values[i] == values[i + 3] and values[i + 3] == values[i + (2 * 3)] and values[i] == 1) or \
                    (values[i * 3] == values[(i * 3) + 1] and values[(i * 3) + 1] == values[(i * 3) + 2] and values[i * 3] == 1) or \
                    (values[i] == values[4] and values[4] == values[8 - i] and values[i] == 1):
                return 1.0
            elif (values[i] == values[i + 3] and values[i + 3] == values[i + (2 * 3)] and values[i] == 2) or \
                    (values[i * 3] == values[(i * 3) + 1] and values[(i * 3) + 1] == values[(i * 3) + 2] and values[i * 3] == 2) or \
                    (values[i] == values[4] and values[4] == values[8 - i] and values[i] == 2):
                return -1.0
        return 0.0

    def winrate(self):
        print(f'win rate : {self.win_rate/self.nterminal if self.nterminal > 0 else 0}')
        print(f'game played : {self.nterminal}')

    def reset(self):
        self.current_cell = 0
        self.step_count = 0

    def view(self):
        print(f'Game Over: {self.is_game_over()}')
        values = self.values_cases()
        for i in range(3):
            for j in range(3):
                if values[(i * 3) + j] == 0:
                    print("_", end="")
                elif values[(i * 3) + j] == 1:
                    print("X", end="")
                else:
                    print("O", end="")
            print()
        if self.is_game_over():
            print(f'score : {self.score()}')
            self.win_rate += self.score()
            self.nterminal += 1
            self.winrate()

    def reset_random(self):
        self.reset()
        aa = self.available_actions_ids()
        while len(aa) > 0:
            random_move = random.randint(0, len(aa) - 1)
            rand = random.randint(0, 1)
            if self.step_count % 2 == 0:
                self.current_cell += rand * 3 ** (8 - aa[random_move])
            else:
                self.current_cell += rand * 2 * (3 ** (8 - aa[random_move]))
            self.step_count += rand
            aa = np.delete(aa, random_move)

    def values_cases(self) -> np.ndarray:
        values = np.zeros(9)
        tmp_state = self.current_cell
        for i in range(9):
            values[i] = tmp_state // 3**(8 - i)
            tmp_state %= 3**(8 - i)
        return values