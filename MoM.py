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
    best_arm = np.argmax(np.sum(cur_contexts, axis=1))
    return cur_contexts, best_arm


# Global: Student t's df
def get_payoff(selected_context, theta, r):
    cur_payoff = np.dot(selected_context, theta) + np.random.standard_t(df, (1, r))
    return cur_payoff


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


# Global: delta
# Implicit: theta = np.ones(shape=(d, 1)) / np.sqrt(d)
def mom(round, d, K, theta, epsilon, v):
    m = round ** (2 * epsilon / (1 + 3 * epsilon))
    a, inverse_a = np.eye(d), np.eye(d)
    b = np.zeros((d, 1))
    hat_theta = np.zeros((d, 1))
    beta = 1
    t, r = 0, int(round / m)
    k = int(8 * np.log(2 / delta))
    n = int(r / k)
    # print(k, n, r)
    cul_regret = 0
    with open("./data/mom_" + str(round) + "_student_" + str(df) + "_noise.txt", "w") as f,\
        open("./data/mom_" + str(round) + "_student_" + str(df) + "_noise_e.txt", "w") as g:
        while t < m:
            print("round", t)
            cur_contexts, best_arm = get_contexts(K, d)
            # print(np.linalg.norm(hat_theta - theta))
            estimated_payoff = np.dot(cur_contexts, hat_theta)  # K*1
            width = []
            for i in range(K):
                width.append(beta * np.sqrt(np.dot(np.dot(cur_contexts[i], inverse_a), cur_contexts[i].T)))
            width = np.array(width).reshape(K, 1)
            estimated_payoff += width
            selected_arm = int(np.argmax(estimated_payoff))
            for _ in range(r):
                cul_regret += np.dot(cur_contexts[best_arm] - cur_contexts[selected_arm], theta)
                f.write(str(cul_regret[0]) + '\t')
                g.write(str(np.linalg.norm(hat_theta - theta)) + '\t')
            t += 1
            cur_payoff = get_payoff(cur_contexts[selected_arm], theta, r)
            cur_payoff = cur_payoff[:, :k * n].reshape(k, n)
            cur_payoff = np.mean(cur_payoff, axis=1, keepdims=True)
            b += cur_contexts[selected_arm].reshape(d, 1) * np.median(cur_payoff)
            a += np.dot(cur_contexts[selected_arm].reshape(d, 1), cur_contexts[selected_arm].reshape(1, d))
            inverse_a = np.linalg.inv(a)
            hat_theta = np.dot(inverse_a, b)
            beta = (12 * v) ** (1 / (1 + epsilon)) * (8 * np.log(2 * m / delta) / r) ** (epsilon / (1 + epsilon)) * (
                    np.sqrt(2 * np.log(m * np.linalg.det(a) ** (1 / 2))) + np.sqrt(t + 1)) + 1
            beta /= 40
            # beta = 2


# Global: delta
# Implicit: theta = np.ones(shape=(d, 1)) / np.sqrt(d)
def mom_mom(round, d, K, theta, epsilon_mom, v_mom, varepsilon, n_tilde):
    
    round_total = round
    round = round_total // n_tilde + 1
    
    m = round ** (2 * epsilon_mom / (1 + 3 * epsilon_mom))
    a, inverse_a = np.eye(d), np.eye(d)
    b = np.zeros((d, 1))
    hat_theta = np.zeros((d, 1))
    beta = 1
    t, r = 0, int(round / m)
    k = int(8 * np.log(2 / delta))
    n = int(r / k)
    # print(k, n, r)
    cul_regret = 0
    with open("./data/mom_mom_" + str(round_total) + "_student_" + str(df) + "_noise.txt", "w") as f,\
        open("./data/mom_mom_" + str(round_total) + "_student_" + str(df) + "_noise_e.txt", "w") as g:
        while t < m:
            print("round", t)
            cur_contexts, best_arm = get_contexts(K, d)
            # print(np.linalg.norm(hat_theta - theta))
            estimated_payoff = np.dot(cur_contexts, hat_theta)  # K*1
            width = []
            for i in range(K):
                width.append(beta * np.sqrt(np.dot(np.dot(cur_contexts[i], inverse_a), cur_contexts[i].T)))
            width = np.array(width).reshape(K, 1)
            estimated_payoff += width
            selected_arm = int(np.argmax(estimated_payoff))
            
            for _ in range(r * n_tilde):
                cul_regret += np.dot(cur_contexts[best_arm] - cur_contexts[selected_arm], theta)
                f.write(str(cul_regret[0]) + '\t')
                g.write(str(np.linalg.norm(hat_theta - theta)) + '\t')
            
            t += 1
            
            cur_payoff = get_payoff_mom(cur_contexts[selected_arm], theta, r, varepsilon, n_tilde)
            
            cur_payoff = cur_payoff[:, :k * n].reshape(k, n)
            cur_payoff = np.mean(cur_payoff, axis=1, keepdims=True)
            b += cur_contexts[selected_arm].reshape(d, 1) * np.median(cur_payoff)
            a += np.dot(cur_contexts[selected_arm].reshape(d, 1), cur_contexts[selected_arm].reshape(1, d))
            inverse_a = np.linalg.inv(a)
            hat_theta = np.dot(inverse_a, b)
            beta = (12 * v_mom) ** (1 / (1 + epsilon_mom)) * (8 * np.log(2 * m / delta) / r) ** (epsilon_mom / (1 + epsilon_mom)) * (
                    np.sqrt(2 * np.log(m * np.linalg.det(a) ** (1 / 2))) + np.sqrt(t + 1)) + 1
            beta /= 40
            # beta = 2


if __name__ == "__main__":
    
    varepsilon = 0.5
    # n_tilde = (16 * np.log(2 * round / delta))**(1/varepsilon)
    n_tilde = 64
    if df > 1:
        # heavy-tailed
        epsilon = (df - 1) / 2
        epsilon_mom = 1
        v_mom = 1 / 500
        v = t(df).expect(lambda x: abs(x) ** (1 + epsilon))
    else:
        # super heavy-tailed
        epsilon = 1
        epsilon_mom = 1
        v_mom = 0.3
        v = v_mom * n_tilde ** (1-varepsilon)
    
    round, K, d = 1000000, 20, 10
    theta = np.ones(shape=(d, 1)) / np.sqrt(d)
    
    mom(round, d, K, theta, epsilon, v)
    mom_mom(round, d, K, theta, epsilon_mom, v_mom, varepsilon, n_tilde)
    

    # draw
    x = np.arange(round)
    plt.figure(figsize=(10, 10), dpi=100)

    # regret
    plt.subplot(2, 1, 1)

    with open("./data/mom_" + str(round) + "_student_" + str(df) + "_noise.txt", "r") as f:
        y_mom = list(map(float, f.readline().strip().split()))
    plt.plot(x[0:round:(round//200)], y_mom[0:round:(round//200)], label="MoM", color="darkblue", linestyle="-", linewidth='3')

    with open("./data/mom_mom_" + str(round) + "_student_" + str(df) + "_noise.txt", "r") as f:
        y_mom_mom = list(map(float, f.readline().strip().split()))
    plt.plot(x[0:round:(round//200)], y_mom_mom[0:round:(round//200)], label="MoM_mom", color="springgreen", linestyle="-", linewidth='3')

    plt.xlabel("Iteration", fontsize=22)
    plt.ylabel("Regret", fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=16)
    plt.grid(linestyle=":")

    # error
    plt.subplot(2, 1, 2)

    with open("./data/mom_" + str(round) + "_student_" + str(df) + "_noise_e.txt", "r") as f:
        y_mom = list(map(float, f.readline().strip().split()))
    plt.plot(x[0:round:(round//200)], y_mom[0:round:(round//200)], label="MoM", color="darkblue", linestyle="-", linewidth='3')

    with open("./data/mom_mom_" + str(round) + "_student_" + str(df) + "_noise_e.txt", "r") as f:
        y_mom_mom = list(map(float, f.readline().strip().split()))
    plt.plot(x[0:round:(round//200)], y_mom_mom[0:round:(round//200)], label="MoM_mom", color="springgreen", linestyle="-", linewidth='3')

    # plt.yscale('log')
    plt.xlabel("Iteration", fontsize=22)
    plt.ylabel("Estimition Error", fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=16)
    plt.grid(linestyle=":")

    plt.savefig("./figures/mom_" + str(round) + "_student_" + str(df).replace('.', '_') + "_noise_" + str(varepsilon) + "_" + str(n_tilde) + ".pdf")
    plt.show()
