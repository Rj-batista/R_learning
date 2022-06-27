from MC.drl_sample_project_python.drl_lib.do_not_touch.mdp_env_wrapper import Env1
from MC.drl_sample_project_python.drl_lib.to_do.GridWorld import *
from MC.drl_sample_project_python.drl_lib.do_not_touch.result_structures import ValueFunction, PolicyAndValueFunction

import numpy as np

from MC.drl_sample_project_python.drl_lib.to_do.GridWorld import LineWorld, GridWorld


def policy_evaluation_on_line_world(pi: np.ndarray) -> ValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    S = LineWorld().states()
    A = LineWorld().actions()
    R = LineWorld().rewards()
    theta = 0.0000001
    V = np.random.random((len(S),))
    V[0] = 0.0
    V[S[-1]] = 0.0

    while True:
        delta = 0
        for s in S:
            v = V[s]
            V[s] = 0.0
            for a in A:
                total = 0.0
                for s_p in S:
                    for r in range(len(R)):
                        total += LineWorld().transition_probability(s, a, s_p, r) * (R[r] + 0.999 * V[s_p])
                total *= pi[s, a]
                V[s] += total
            delta = max(delta, np.abs(v - V[s]))
        if delta < theta:
            break
    return V


def policy_iteration_on_line_world(pi: np.ndarray) -> PolicyAndValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """

    S = LineWorld().states()
    A = LineWorld().actions()
    R = LineWorld().rewards()
    theta = 0.0000001
    V = np.random.random((len(S),))
    V[0] = 0.0
    V[len(S) - 1] = 0.0

    pi = np.random.random((len(S), (len(A))))
    for s in S:
        pi[s] = np.sum(pi[s])

    pi[0] = 0.0
    pi[len(S) - 1] = 0.0
    print('Initial policy : ', pi)

    while True:
        # policy evaluation
        while True:
            delta = 0
            for s in S:
                v = V[s]
                V[s] = 0.0
                for a in A:
                    total = 0.0
                    for s_p in S:
                        for r in range(len(R)):
                            total += round(LineWorld().transition_probability(s, a, s_p, r) * (R[r] + 0.999 * V[s_p]),4)
                    total *= pi[s, a]
                    V[s] += total
                delta = max(delta, np.abs(v - V[s]))
            if delta < theta:
                break

        # policy improvement
        stable = True
        for s in S:
            old_pi_s = pi[s].copy()
            best_a = -1
            best_a_score = -99999999999
            for a in A:
                total = 0
                for s_p in S:
                    for r in range(len(R)):
                        total += LineWorld().transition_probability(s, a, s_p, r) * (R[r] + 0.999 * V[s_p])
                if total > best_a_score:
                    best_a = a
                    best_a_score = total
            pi[s, :] = 0.0
            pi[s, best_a] = 1.0
            if np.any(pi[s] != old_pi_s):
                stable = False
        if stable:
            return pi, V


def value_iteration_on_line_world(pi: np.ndarray) -> PolicyAndValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    S = LineWorld().states()
    A = LineWorld().actions()
    R = LineWorld().rewards()
    theta = 0.0000001
    V = np.random.random((len(S),))
    V[0] = 0.0
    V[S[-1]] = 0.0

    while True:
        delta = 0
        for s in S:
            v = V[s]  # Old_value = v
            V[s] = 0.0
            for a in A:
                total = 0.0
                for s_p in S:
                    new_value = 0
                    for r in range(len(R)):
                        total += LineWorld().transition_probability(s, a, s_p, r) * (R[r] + 0.999 * V[s_p])
                total *= pi[s, a]
                V[s] += total
                if total > new_value:
                    new_value = total
            V[s] = new_value
            delta = max(delta, np.abs(v - V[s]))
        if delta < theta:
            break
    print(V)
    return V


def policy_evaluation_on_grid_world(pi: np.ndarray) -> ValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    S = GridWorld().states()
    A = GridWorld().actions()
    R = GridWorld().rewards()

    theta = 0.0000001
    V = np.random.random((len(S),))
    V[0] = 0.0
    V[S[-1]] = 0.0

    while True:
        delta = 0
        for s in S:
            v = V[s]
            V[s] = 0.0
            for a in A:
                total = 0.0
                for s_p in S:
                    for r in range(len(R)):
                        total += GridWorld().transition_probability(s, a, s_p, r) * (R[r] + 0.999 * V[s_p])
                total *= pi[s, a]
                V[s] += total
            delta = max(delta, np.abs(v - V[s]))
        if delta < theta:
            break
    return V

def policy_iteration_on_grid_world() -> PolicyAndValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    # TODO
    pass


def value_iteration_on_grid_world() -> PolicyAndValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    # TODO
    pass


def policy_evaluation_on_secret_env1() -> ValueFunction:
    """
    Creates a Secret Env1
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    env = Env1()
    # TODO
    pass


def policy_iteration_on_secret_env1() -> PolicyAndValueFunction:
    """
    Creates a Secret Env1
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    env = Env1()
    # TODO
    pass


def value_iteration_on_secret_env1() -> PolicyAndValueFunction:
    """
    Creates a Secret Env1
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Prints the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    env = Env1()
    # TODO
    pass


def demo():

    S = LineWorld().states()
    A = LineWorld().actions()
    right_pi = np.zeros((len(S), len(A)))
    right_pi[:, 1] = 1.0

    print(policy_evaluation_on_line_world(right_pi))
    print(policy_iteration_on_line_world(right_pi))
    print(value_iteration_on_line_world(right_pi))

    S = GridWorld().states()
    A = GridWorld().actions()
    pi = np.ones((len(S), len(A))) * 0.5
    print(policy_evaluation_on_grid_world(pi))
    """print(policy_iteration_on_grid_world())
    print(value_iteration_on_grid_world())

    print(policy_evaluation_on_secret_env1())
    print(policy_iteration_on_secret_env1())
    print(value_iteration_on_secret_env1())"""
