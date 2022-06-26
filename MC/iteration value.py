import numpy as np

S = [0, 1, 2, 3, 4]
A = [0, 1]
R = [-1.0, 0.0, 1.0]

p = np.zeros((len(S), len(A), len(S), len(R)))


for s in S[1:-1]:
    if s == 1:
        p[s, 0, s - 1, 0] = 1.0
    else:
        p[s, 0, s - 1, 1] = 1.0

    if s == 3:
        p[s, 1, s + 1, 2] = 1.0
    else:
        p[s, 1, s + 1, 1] = 1.0

right_pi = np.zeros((len(S), len(A)))
right_pi[:, 1] = 1.0

left_pi = np.zeros((len(S), len(A)))
left_pi[:, 0] = 1.0

random_pi = np.ones((len(S), len(A))) * 0.5

def iteration_value(pi):
    theta = 0.0000001
    V = np.random.random((len(S),))
    V[0] = 0.0
    V[S[-1]] = 0.0

    while True:
        delta = 0
        for s in S:
            v = V[s] #Old_value = v
            V[s] = 0.0
            for a in A:
                total = 0.0
                for s_p in S:
                    new_value=0
                    for r in range(len(R)):
                        total += p[s, a, s_p, r] * (R[r] + 0.999 * V[s_p])
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


iteration_value(left_pi)
iteration_value(right_pi)
iteration_value(random_pi)