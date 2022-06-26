import itertools
import random

import numpy as np
import numba

S = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
A = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
R = np.array([-1.0, 0.0, 1.0])


victoire = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]
all_victoire = []
for possibility in victoire:
    res = [list(ele) for ele in itertools.permutations(possibility)]
    all_victoire.extend(res)
all_victoire



def step(s: int, a: int) -> (int, float, bool):
    combination = all_victoire
    S = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    humain_s = []
    humain_a = []
    ordi_a = []
    ordi_s = []
    ordi_a.append(a)
    ordi_s.append(s)
    for value in ordi_s:
        for item in combination:
            if value in item:
                combination.remove(item)


    play_available = list(set(S) - set(humain_s) - set(ordi_s))
    next_play = random.choice(random.choice(play_available))
    humain_a.append(next_play)
    humain_s.append(next_play)
    play_available = list(set(S) - set(humain_s)-set(ordi_s))

    all_combination_ordi = [list(ele) for ele in itertools.permutations(ordi_s, 3)]
    all_combination_humain = [list(ele) for ele in itertools.permutations(humain_s, 3)]
    if any(True for x in all_combination_ordi if x in all_victoire) == False:
        return next_play, 0, False, play_available
    if any(True for x in all_combination_humain if x in all_victoire) == True:
        return next_play, -1.0, True
    if any(True for x in all_combination_ordi if x in all_victoire) == True:
        return next_play, 1, True









# @numba.jit(nopython=True, parallel=True)
def monte_carlo_control_with_exploring_starts(S, A, iter_count):
    pi = np.random.random((len(S), len(A)))
    for s in S:
        pi[s] /= np.sum(pi[s])

    q = np.random.random((len(S), len(A)))

    Returns = [[[] for a in A] for s in S]

    for it in range(iter_count):
        first_choice = np.random.choice(S)
        s0 = first_choice #l'ordi joue une position
        a0 = first_choice
        s_ordi = s0
        a_ordi = a0

        _, _, terminal_ordi, play_av = step(s_ordi, a_ordi) #on récupère la valeur du prochain coup
        s_history = [s_ordi]
        a_history = [s_ordi]
        r_history = []
        r_history.append(0)
        while terminal_ordi == False and len(play_av)>0:
            choice = np.random.choice(play_av)
            s_ordi = choice
            a_ordi = choice

            _, r, terminal,play_av = step(a_ordi, a_ordi)
            s_history.append(s_ordi)
            a_history.append(a_ordi)
            r_history.append(r)

        G = 0
        for t in reversed(range(len(s_history))):
            G = 0.999 * G + r_history[t]
            s_t = s_history[t]
            a_t = a_history[t]

            appear = False
            for t_p in range(t - 1):
                if s_history[t_p] == s_t and a_history[t_p] == a_t:
                    appear = True
                    break
            if appear:
                continue

            Returns[s_t][a_t].append(G)
            q[s_t, a_t] = np.mean(Returns[s_t][a_t])
            pi[s_t, :] = 0.0
            pi[s_t, np.argmax(q[s_t])] = 1.0

    return pi, q


print(monte_carlo_control_with_exploring_starts(S, A, 10000))
