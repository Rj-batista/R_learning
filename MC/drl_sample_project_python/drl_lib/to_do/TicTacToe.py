import random
import numpy as np
from MC.drl_sample_project_python.drl_lib.do_not_touch.contracts import SingleAgentEnv


class TicTacToe(SingleAgentEnv):
    def __init__(self):
        self.nb_cells = 9
        self.current_cell = 0
        self.step_count = 0
        self.nterminal = 0
        self.theBoard = [" ", " ", " ", " ", " ", " ", " ", " ", " "]

    def state_id(self) -> int:
        return self.current_cell

    def take_turn(self):
        if self.step_count % 2 == 0:
            return "X"
        else :
            return "O"

    def is_game_over(self) -> bool:
        if self.step_count == 8:
            return True
        elif self.theBoard[6] == self.theBoard[7] == self.theBoard[8] != ' ':  # --- top
            return True
        elif self.theBoard[3] == self.theBoard[4] == self.theBoard[5] != ' ':  # --- middle
            return True
        elif self.theBoard[0] == self.theBoard[1] == self.theBoard[2] != ' ':  # --- bottom
            return True
        elif self.theBoard[0] == self.theBoard[3] == self.theBoard[6] != ' ':  # | left side
            return True
        elif self.theBoard[1] == self.theBoard[4] == self.theBoard[7] != ' ':  # | middle
            return True
        elif self.theBoard[2] == self.theBoard[5] == self.theBoard[8] != ' ':  # | right side
            return True
        elif self.theBoard[6] == self.theBoard[4] == self.theBoard[2] != ' ':  # / diagonal
            return True
        elif self.theBoard[0] == self.theBoard[4] == self.theBoard[8] != ' ':  # \ diagonal
            return True
        else:
            return False

    def available_actions_ids(self) -> np.ndarray:
        available = []
        for i in range(len(self.theBoard)):
            if self.theBoard[i] == " ":
                available.append(i)
        return available

    def act_with_action_id(self, action_id: int):
        if self.theBoard[action_id] == " ":
            if self.take_turn() == "X":
                self.theBoard[action_id] = "X"
            elif self.take_turn() == "O":
                self.theBoard[action_id] = "O"
            self.step_count += 1
        else:
            print("La case est occupé")


    def score(self) -> float:
        turn = self.take_turn()
        if turn == "X":
            if (self.theBoard[6] == self.theBoard[7] == self.theBoard[8] != ' ') or (
                    self.theBoard[3] == self.theBoard[4] == self.theBoard[5] != ' ') or (
                    self.theBoard[0] == self.theBoard[1] == self.theBoard[2] != ' ') or (
                    self.theBoard[0] == self.theBoard[3] == self.theBoard[6] != ' ') or (
                    self.theBoard[1] == self.theBoard[4] == self.theBoard[7] != ' ') or (
                    self.theBoard[2] == self.theBoard[5] == self.theBoard[8] != ' ') or (
                    self.theBoard[6] == self.theBoard[4] == self.theBoard[2] != ' ') or (
                    self.theBoard[0] == self.theBoard[4] == self.theBoard[8] != ' '):
                return -1.0
        if turn == "O":
            if (self.theBoard[6] == self.theBoard[7] == self.theBoard[8] != ' ') or (
                    self.theBoard[3] == self.theBoard[4] == self.theBoard[5] != ' ') or (
                    self.theBoard[0] == self.theBoard[1] == self.theBoard[2] != ' ') or (
                    self.theBoard[0] == self.theBoard[3] == self.theBoard[6] != ' ') or (
                    self.theBoard[1] == self.theBoard[4] == self.theBoard[7] != ' ') or (
                    self.theBoard[2] == self.theBoard[5] == self.theBoard[8] != ' ') or (
                    self.theBoard[6] == self.theBoard[4] == self.theBoard[2] != ' ') or (
                    self.theBoard[0] == self.theBoard[4] == self.theBoard[8] != ' '):
                return 1.0
        else:
            return 0.0

    def reset(self):
        self.current_cell = 0
        self.step_count = 0
        self.theBoard = [" ", " ", " ", " ", " ", " ", " ", " "]

    def view(self):
        print(self.theBoard[0] + '|' + self.theBoard[1] + '|' + self.theBoard[2])
        print('-+-+-')
        print(self.theBoard[3] + '|' + self.theBoard[4] + '|' + self.theBoard[5])
        print('-+-+-')
        print(self.theBoard[6] + '|' + self.theBoard[7] + '|' + self.theBoard[8])

