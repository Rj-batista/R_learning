import numpy as np
'''
L'environnement est définit par les variable S(state) A(action) R(reward)
Nous avons donc 5 états avec des rewards sur les positions 0 et 4
Nous avons deux actions 0 et 1 pour aller à gauche et aller à droite
La reward est -1 pour la position S=0 et 1 S=4
'''
S = [0, 1, 2, 3, 4]
A = [0, 1]
R = [-1.0, 0.0, 1.0]
# matrice de probablité de chaque action (règle du jeu)
p = np.zeros((len(S), len(A), len(S), len(R)))

'''
On définit en dessous les déplacements et reward associés 
la probabilité de 1 avec l'action 0 nous amène à l'état 0 et nous donne la récompense -1 et la probabilité est égale à 1
1er valeur ou je suis ensuite ce que je fais et ensuite les conséquences  
'''
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


def policy_evaluation(pi: np.ndarray):
    theta = 0.0000001
    V = np.random.random((len(S),))
    V[0] = 0.0
    V[4] = 0.0

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
    print(V)
    return V


policy_evaluation(left_pi)
policy_evaluation(right_pi)
policy_evaluation(random_pi)
