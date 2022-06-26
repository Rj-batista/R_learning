import numpy as np
import numba

nb_cells = 500

S = np.arange(nb_cells)
A = np.array([0, 1])
R = np.array([-1.0, 0.0, 1.0])

p = np.zeros((len(S), len(A), len(S), len(R)))

for s in S[1:-1]:
    if s == 1:
        p[s, 0, s - 1, 0] = 1.0
    else:
        p[s, 0, s - 1, 1] = 1.0

    if s == nb_cells - 2:
        p[s, 1, s + 1, 2] = 1.0
    else:
        p[s, 1, s + 1, 1] = 1.0


@numba.jit(nopython=True, parallel=True)
def policy_iteration(S, A, R, p):
    theta = 0.0000001
    V = np.random.random((len(S),))
    V[0] = 0.0
    V[nb_cells - 1] = 0.0

    pi = np.random.random((len(S), (len(A))))
    for s in S:
        pi[s] /= np.sum(pi[s])

    pi[0] = 0.0
    pi[nb_cells - 1] = 0.0
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
                            total += p[s, a, s_p, r] * (R[r] + 0.999 * V[s_p])
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
                        total += p[s, a, s_p, r] * (R[r] + 0.999 * V[s_p])
                if total > best_a_score:
                    best_a = a
                    best_a_score = total
            pi[s, :] = 0.0
            pi[s, best_a] = 1.0
            if np.any(pi[s] != old_pi_s):
                stable = False
        if stable:
            return pi, V


print(policy_iteration(S, A, R, p))
