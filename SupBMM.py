"""
This code is originally from the paper https://arxiv.org/abs/2004.13465 with permission,
and was modified according to our needs.
"""

import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt


delta = 0.01
df = 3



# Implicit: theta = np.ones(shape=(d, 1)) / np.sqrt(d)
def get_contexts(K, d):
    cur_contexts = []
    for _ in range(K):
        cur_contexts.append(np.random.uniform(size=d))
    cur_contexts = np.array(cur_contexts)  # K*d
    x_norm = np.linalg.norm(cur_contexts, ord=2, axis=1, keepdims=True)  # K*1
    cur_contexts = cur_contexts / x_norm
    best_arm = int(np.argmax(np.sum(cur_contexts, axis=1)))
    return cur_contexts, best_arm


# Global: Student t's df
def get_payoff(selected_context, theta, r):
    cur_payoff = np.dot(selected_context, theta) + np.random.standard_t(df, (1, r))
    return cur_payoff  # 1*r


def mean_of_medians(payoff, varepsilon, n_tilde):
    """
    Input: payoff (n_tilde, r)
    Use mean-of-medians subroutine to process every column of the raw payoff.
    Output: payoff (1, r)
    """
    # assert(payoff.shape[0] == n_tilde)
    k = int(n_tilde ** varepsilon)
    k_ = n_tilde // k
    r = payoff.shape[1]
    payoff = np.resize(payoff, (k, k_, r))
    payoff = np.median(payoff, axis=0)
    return np.mean(payoff, axis=0, keepdims=True)


# Global: Student t's df
def get_payoff_mom(selected_context, theta, r, varepsilon, n_tilde):
    cur_payoff = np.dot(selected_context, theta) + np.random.standard_t(df, (n_tilde, r))  # n_tilde*r
    cur_payoff = mean_of_medians(cur_payoff, varepsilon, n_tilde)
    return cur_payoff  # 1*r


# Implicit: theta = np.ones(shape=(d, 1)) / np.sqrt(d)
def bmm(psi, historical_information, historical_payoff, cur_contexts, d, num_arms, epsilon, v):
    if psi == 0:
        alpha = 0.01 / 6 * (12 * v) ** (1 / (1 + epsilon))
        width = []
        for i in range(num_arms):
            width.append(alpha * np.sqrt(np.dot(cur_contexts[i], cur_contexts[i].T)))
        return np.array([0] * num_arms).reshape(num_arms, 1), np.array(width).reshape(num_arms, 1)
    a = np.eye(d)
    for i in range(psi):
        a += np.dot(historical_information[i].reshape(d, 1), historical_information[i].reshape(1, d))  # d*d
    b = np.dot(historical_information.T, historical_payoff)  # d*r
    inverse_a = np.linalg.inv(a)
    hat_theta = np.dot(inverse_a, b)  # d*r
    estimated_payoffs = np.dot(cur_contexts, hat_theta)  # K*r
    estimated_payoff = np.median(estimated_payoffs, 1)  # K*1
    alpha = 0.01 / 6 * (12 * v) ** (1 / (1 + epsilon)) * psi ** ((1 - epsilon) / (2 * (1 + epsilon)))
    width = []
    for i in range(num_arms):
        width.append(alpha * np.sqrt(np.dot(np.dot(cur_contexts[i], inverse_a), cur_contexts[i].T)))
    # print(np.linalg.norm(hat_theta[:, 0] - np.ones(shape=(d, 1)) / np.sqrt(d)))
    return estimated_payoff.reshape(num_arms, 1), np.array(width).reshape(num_arms, 1)


# Global: delta
def supbmm(round, d, K, theta, epsilon, v):
    t = 0
    # r = int(8 * np.log(2 * round * K * np.log(round) / delta))
    r = 31
    cur_round = round // r + 1
    historical_information = np.zeros((cur_round, d))
    historical_payoff = np.zeros((cur_round, r))
    cul_regret = 0

    with open("./data/supbmm_" + str(round) + "_student_" + str(df) + "_noise.txt", "w") as f:
        while t < cur_round:
            print("round", t)
            cur_contexts, best_arm = get_contexts(K, d)
            estimated_payoff, width = bmm(t, historical_information, historical_payoff, cur_contexts, d, K, epsilon, v)
            selected_arm = int(np.argmax(estimated_payoff + width))
            selected_context = cur_contexts[selected_arm]
            cur_payoff = get_payoff(selected_context, theta, r)
            historical_payoff[t] = cur_payoff
            historical_information[t] = selected_context
            for i in range(r):
                cul_regret += np.dot(cur_contexts[best_arm] - selected_context, theta)
                f.write(str(cul_regret[0]) + '\t')
            t += 1


# Global: delta
def supbmm_mom(round, d, K, theta, epsilon_mom, v_mom, varepsilon, n_tilde):
    
    round_total = round
    round = round_total // n_tilde + 1
    
    t = 0
    # r = int(8 * np.log(2 * round * K * np.log(round) / delta))
    r = 31
    cur_round = round // r + 1
    historical_information = np.zeros((cur_round, d))
    historical_payoff = np.zeros((cur_round, r))
    cul_regret = 0

    with open("./data/supbmm_mom_" + str(round_total) + "_student_" + str(df) + "_noise.txt", "w") as f:
        while t < cur_round:
            print("round", t)
            cur_contexts, best_arm = get_contexts(K, d)
            estimated_payoff, width = bmm(t, historical_information, historical_payoff, cur_contexts, d, K, epsilon_mom, v_mom)
            selected_arm = int(np.argmax(estimated_payoff + width))
            selected_context = cur_contexts[selected_arm]
            
            cur_payoff = get_payoff_mom(selected_context, theta, r, varepsilon, n_tilde)
            
            historical_payoff[t] = cur_payoff
            historical_information[t] = selected_context
            
            for i in range(r * n_tilde):
                cul_regret += np.dot(cur_contexts[best_arm] - selected_context, theta)
                f.write(str(cul_regret[0]) + '\t')
            
            t += 1


if __name__ == "__main__":
    
    varepsilon = 0.5
    # n_tilde = (16 * np.log(2 * round / delta))**(1/varepsilon)
    n_tilde = 64
    if df > 1:
        # heavy-tailed
        epsilon = (df - 1) / 2
        epsilon_mom = 1
        v_mom = 1
        v = t(df).expect(lambda x: abs(x) ** (1 + epsilon))
    else:
        # super heavy-tailed
        epsilon = 1
        epsilon_mom = 1
        v_mom = 3
        v = v_mom * n_tilde ** (1-varepsilon)
    
    round, K, d = 100000, 20, 10
    theta = np.ones(shape=(d, 1)) / np.sqrt(d)
    
    supbmm(round, d, K, theta, epsilon, v)
    supbmm_mom(round, d, K, theta, epsilon_mom, v_mom, varepsilon, n_tilde)
    

    # draw
    x = np.arange(round)
    plt.figure(figsize=(10, 6), dpi=100)

    with open("./data/supbmm_" + str(round) + "_student_" + str(df) + "_noise.txt", "r") as f:
        y_bmm = list(map(float, f.readline().strip().split()))
    plt.plot(x[0:round:(round//200)], y_bmm[0:round:(round//200)], label="SupBMM", color="darkblue", linestyle="-", linewidth='3')

    with open("./data/supbmm_mom_" + str(round) + "_student_" + str(df) + "_noise.txt", "r") as f:
        y_bmm_mom = list(map(float, f.readline().strip().split()))
    plt.plot(x[0:round:(round//200)], y_bmm_mom[0:round:(round//200)], label="SupBMM_mom", color="springgreen", linestyle="-", linewidth='3')

    plt.xlabel("Iteration", fontsize=22)
    plt.ylabel("Regret", fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=16)
    plt.grid(linestyle=":")
    plt.savefig("./figures/supbmm_" + str(round) + "_student_" + str(df).replace('.', '_') + "_noise_" + str(varepsilon) + "_" + str(n_tilde) + ".pdf")
    plt.show()
