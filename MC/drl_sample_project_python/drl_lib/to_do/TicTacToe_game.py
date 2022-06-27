from MC.drl_sample_project_python.drl_lib.to_do import TicTacToe
from MC.drl_sample_project_python.drl_lib.to_do.monte_carlo_methods import *



def game():
    tic = TicTacToe()
    while tic.is_game_over(tic.theBoard) == False:
        tic.view()
        choix = input("Faite un choix entre 0 et 8:")
        tic.act_with_action_id(choix)
        tic.step_count += 1
        tic.view()
        monte_carlo_es_on_tic_tac_toe_solo.monte_carlo_control_with_exploring_starts(tic.available_actions_ids()
                                                                                     ,tic.available_actions_ids())
