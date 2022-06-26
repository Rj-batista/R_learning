from RlearningMDPenvimplementation.MC.drl_sample_project_python.drl_lib.do_not_touch.contracts import MDPEnv, SingleAgentEnv
import numpy as np
import random


class LineWorld(MDPEnv):
    def __init__(self):
        self.current_cell = random.randint(0, 6)
        self.nb_cells = 7
        self.terminal_values = [0, 6]
    def states(self) -> np.ndarray:
        return np.arange(self.nb_cells)

    def actions(self) -> np.ndarray:
        return np.arange(2)

    def rewards(self) -> np.ndarray:
        return np.array([-1.0, 0.0,1.0])

    def is_state_terminal(self, s: int) -> bool:
        if s in self.terminal_values:
            return True
        return False

    def transition_probability(self, s: int, a: int, s_p: int, r: float) -> float:
        p = self.p()
        return np.round(p[s, a, s_p, r],5)

    def p(self):
        S, A, R = self.states(), self.actions(), self.rewards()
        p = np.zeros((len(S), len(A), len(S), len(R)),dtype=np.longfloat)

        for s in S[1:-1]:
            if s == 1:
                p[s, 0, s - 1, 0] = 1.0
            else:
                p[s, 0, s - 1, 1] = 1.0

            if s == S[-2]:
                p[s, 1, s + 1, 2] = 1.0
            else:
                p[s, 1, s + 1, 1] = 1.0
        return p

    def view_state(self, s: int):
        for i in range(len(self.states())):
            if i == s:
                print('X', end=' ')
            else:
                print('-', end=' ')


class GridWorld(MDPEnv):
    def __init__(self):
        random.seed(42)
        self.current_cell = random.randint(0, 24)
        self.cols, self.rows = 5, 5
        self.nb_cells = 25
        self.terminal_values = [4, 24]
        
    
    def states(self) -> np.ndarray:
        return np.arange(self.nb_cells)

    def actions(self) -> np.ndarray:
        return np.array([0,1,2,3])

    def rewards(self) -> np.ndarray:
        return np.array([-3.0, 0.0 , 1.0])

    def is_state_terminal(self, s: int) -> bool:
        if s in self.terminal_values:
            return True
        return False

    def transition_probability(self, s: int, a: int, s_p: int, r: float) -> float:
        p = self.p()
        return np.round(p[s, a, s_p, r],5)

    def p(self):
        S, A, R = self.states(), self.actions(), self.rewards()
        p = np.zeros((len(S), len(A), len(S), len(R)),dtype=np.longfloat)
        haut = [0, 1, 2, 3, 4]
        bas = [20, 21, 22, 23, 24]
        gauche = [0, 5, 10, 15, 20]
        droite = [4, 9, 14, 19, 24]

        for s in S[:-1]:
            # gauche
            if s in gauche:
                p[s, 1, s - 1, 1] = 0.0
            else:
                p[s, 1, s - 1, 1] = 1.0
            # haut
            if s in haut:
                p[s, 2, s - self.rows, 1] = 0.0
            elif s == 9:
                p[s, 2, s - self.rows, 0] = 1.0
            else:
                p[s, 2, s - self.rows, 1] = 1.0
            # droite
            if s in droite:
                p[s, 3, s - 1, 1] = 0.0
            elif s == 4:
                p[s, 3, s + 1, 0] = 1.0
            elif s == 23:
                p[s, 3, s + 1, 2] = 1.0
            else:
                p[s, 3, s + 1, 1] = 1.0
            # bas
            if s in bas:
                p[s, 0, s + self.rows, 1] = 0.0
            if s == 19:
                p[s, 0, s + self.rows, 2] = 1.0
            else:
                p[s, 0, s + self.rows, 1] = 1.0

        return p

    def view_state(self, s: int):
        for i in range(self.nb_cells):
            if i % 5 == 0:
                print('\n')
            if i == s:
                print('X', end=' ')
            else:
                print('-', end=' ')


"""class GridWorld(SingleAgentEnv):
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
        self.current_cell = random.randint(0, self.nb_cells - 1)"""
