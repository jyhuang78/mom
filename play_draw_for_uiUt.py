import numpy as np
import matplotlib.pyplot as plt


delta = 0.01


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


# Implicit: Student t's df
def get_payoff(selected_context, theta, r=1):
    if r == 1:
        cur_payoff = np.dot(selected_context, theta) + np.random.standard_t(df, 1)
    else:
        cur_payoff = np.dot(selected_context, theta) + np.random.standard_t(df, (1, r))
    return cur_payoff


def mean_of_medians(payoff, varepsilon, n_tilde):
    """
    Input: payoff (n_tilde)
    Use mean-of-medians subroutine to process every column of the raw payoff.
    Output: payoff (1)

    or

    Input: payoff (n_tilde, r)
    Use mean-of-medians subroutine to process every column of the raw payoff.
    Output: payoff (1, r)
    """
    k = int(n_tilde ** varepsilon)
    k_ = n_tilde // k
    if len(payoff) == n_tilde:
        payoff = np.resize(payoff, (k, k_))
        payoff = np.median(payoff, axis=0)
        return np.mean(payoff, keepdims=True)
    else:
        r = payoff.shape[1]
        payoff = np.resize(payoff, (k, k_, r))
        payoff = np.median(payoff, axis=0)
        return np.mean(payoff, axis=0, keepdims=True)


# Global: Student t's df
def get_payoff_mom(selected_context, theta, varepsilon, n_tilde, r=1):
    if r == 1:
        cur_payoff = np.dot(selected_context, theta) + np.random.standard_t(df, n_tilde)
        cur_payoff = mean_of_medians(cur_payoff, varepsilon, n_tilde)
    else:
        cur_payoff = np.dot(selected_context, theta) + np.random.standard_t(df, (n_tilde, r))  # n_tilde*r
        cur_payoff = mean_of_medians(cur_payoff, varepsilon, n_tilde)
    return cur_payoff  # 1*1


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


# Implicit: theta = np.ones(shape=(d, 1)) / np.sqrt(d)
# Global: delta
def btc(psi, historical_information, historical_payoff, cur_contexts, d, num_arms, epsilon, v):
    if psi == 0:
        alpha = 0.01 / 3 * v
        width = []
        for i in range(num_arms):
            width.append(alpha * np.sqrt(np.dot(cur_contexts[i], cur_contexts[i].T)))
        return np.array([0] * num_arms).reshape(num_arms, 1), np.array(width).reshape(num_arms, 1)
    a = np.dot(historical_information.T, historical_information) + np.eye(d)  # d*d
    inverse_a = np.linalg.inv(a)
    temp = np.dot(inverse_a, historical_information.T)  # d*t
    estimated_payoff = []
    alpha = 0.01 / 3 * v * psi ** ((1 - epsilon) / (2 * (1 + epsilon)))
    width = []
    for i in range(num_arms):
        beta = np.dot(cur_contexts[i].reshape(1, d), temp)  # 1*t
        beta_norm = np.linalg.norm(beta, ord=2, axis=1, keepdims=True)  # 1*1
        hat_y = np.zeros(shape=(len(historical_payoff), 1))
        for j in range(len(historical_payoff)):
            if abs(historical_payoff[j][0] * beta[0][j]) > beta_norm:
                hat_y[j][0] = 0
            else:
                hat_y[j][0] = historical_payoff[j][0]
        hat_theta = np.dot(np.dot(inverse_a, historical_information.T), hat_y)
        estimated_payoff.append(np.dot(beta, hat_y)[0][0])
        width.append(alpha * np.sqrt(np.dot(np.dot(cur_contexts[i].reshape(d, 1).T, inverse_a), cur_contexts[i].reshape(d, 1))))
    # print(np.linalg.norm(hat_theta - np.ones(shape=(d, 1)) / np.sqrt(d)))
    return np.array(estimated_payoff).reshape(num_arms, 1), np.array(width).reshape(num_arms, 1)


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

    with open("./data/uiUt/supbmm_mom_" + str(round_total) + "_student_" + str(df) + "_noise_" + str(v_mom) + "_path_" + str(i_path) + ".txt", "w") as f:
        while t < cur_round:
            # print("round", t)
            cur_contexts, best_arm = get_contexts(K, d)
            estimated_payoff, width = bmm(t, historical_information, historical_payoff, cur_contexts, d, K, epsilon_mom, v_mom)
            selected_arm = int(np.argmax(estimated_payoff + width))
            selected_context = cur_contexts[selected_arm]
            
            cur_payoff = get_payoff_mom(selected_context, theta, varepsilon, n_tilde, r)
            
            historical_payoff[t] = cur_payoff
            historical_information[t] = selected_context
            
            for i in range(r * n_tilde):
                cul_regret += np.dot(cur_contexts[best_arm] - selected_context, theta)
                f.write(str(cul_regret[0]) + '\t')
            
            t += 1


def supbtc_mom(round, d, K, theta, epsilon_mom, v_mom, varepsilon, n_tilde):
    
    round_total = round
    round = round_total // n_tilde + 1
    
    t = 0
    historical_information = np.zeros((round, d))
    historical_payoff = np.zeros((round, 1))
    cul_regret = 0

    with open("./data/uiUt/supbtc_mom_" + str(round_total) + "_student_" + str(df) + "_noise_" + str(v_mom) + "_path_" + str(i_path) + ".txt", "w") as f:
        while t < round:
            # print("round", t)
            cur_contexts, best_arm = get_contexts(K, d)
            estimated_payoff, width = btc(t, historical_information[:t,:], historical_payoff[:t,:], cur_contexts, d, K, epsilon_mom, v_mom)
            selected_arm = np.argmax(estimated_payoff + width)        
            selected_context = cur_contexts[selected_arm]
            
            cur_payoff = get_payoff_mom(selected_context, theta, varepsilon, n_tilde)  # 1*1
            
            historical_payoff[t] = cur_payoff
            historical_information[t] = selected_context
            
            for _ in range(n_tilde):
                cul_regret += np.dot(cur_contexts[best_arm] - selected_context, theta)
                f.write(str(cul_regret[0]) + '\t')
            
            t += 1


if __name__ == "__main__":
    num_path = 500
    v_mom_list = [0.1, 1, 10]

    df = 0.5

    varepsilon = 0.5
    # n_tilde = (16 * np.log(2 * round / delta))**(1/varepsilon)
    n_tilde_bmm = 256
    n_tilde_btc = 400
    # super heavy-tailed
    epsilon_mom = 1
    # v_mom_bmm = 1
    # v_mom_btc = 1/2

    round, K, d = 10000, 20, 10
    theta = np.ones(shape=(d, 1)) / np.sqrt(d)
    # for v_mom in v_mom_list:
    #     for i_path in range(num_path):
    #         supbmm_mom(round, d, K, theta, epsilon_mom, v_mom, varepsilon, n_tilde_bmm)
    #         supbtc_mom(round, d, K, theta, epsilon_mom, v_mom, varepsilon, n_tilde_btc)
    
    # draw

    x = np.arange(0, round)
    y_bmm = np.zeros((len(v_mom_list), num_path, round))
    y_btc = np.zeros((len(v_mom_list), num_path, round))

    for i_v, v_mom in enumerate(v_mom_list):
        for i_path in range(num_path):
            with open("./data/uiUt/supbmm_mom_" + str(round) + "_student_" + str(df) + "_noise_" + str(v_mom) + "_path_" + str(i_path) + ".txt", "r") as f:
                y_bmm[i_v, i_path] = list(map(float, f.readline().strip().split()))[:round]
            with open("./data/uiUt/supbtc_mom_" + str(round) + "_student_" + str(df) + "_noise_" + str(v_mom) + "_path_" + str(i_path) + ".txt", "r") as f:
                y_btc[i_v, i_path] = list(map(float, f.readline().strip().split()))[:round]
        y_bmm_mean = np.mean(y_bmm, axis=1)
        y_bmm_median = np.median(y_bmm, axis=1)
        y_btc_mean = np.mean(y_btc, axis=1)
        y_btc_median = np.median(y_btc, axis=1)
        
        # SupBMM_mom

        # Mean
        plt.figure(figsize=(10, 6), dpi=100)
        plt.plot(x[0:round:(round//20)], y_bmm_mean[0][0:round:(round//20)], label="v="+str(v_mom_list[0]), color="tab:cyan", linestyle="-", linewidth='3')
        plt.plot(x[0:round:(round//20)], y_bmm_mean[1][0:round:(round//20)], label="v="+str(v_mom_list[1]), color="red", linestyle="-", linewidth='3')
        plt.plot(x[0:round:(round//20)], y_bmm_mean[2][0:round:(round//20)], label="v="+str(v_mom_list[2]), color="green", linestyle="-", linewidth='3')
        plt.xlabel("Iteration", fontsize=22)
        plt.ylabel("Mean Regret", fontsize=22)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend(fontsize=16)
        plt.grid(linestyle=":")
        plt.savefig("./figures/student_" + str(df).replace('.', '_') + "_noise_" + str(round) + "_bmm_mean_uiUt.pdf")
        # plt.show()

        # Median
        plt.figure(figsize=(10, 6), dpi=100)
        plt.plot(x[0:round:(round//20)], y_bmm_median[0][0:round:(round//20)], label="v="+str(v_mom_list[0]), color="tab:cyan", linestyle="-", linewidth='3')
        plt.plot(x[0:round:(round//20)], y_bmm_median[1][0:round:(round//20)], label="v="+str(v_mom_list[1]), color="red", linestyle="-", linewidth='3')
        plt.plot(x[0:round:(round//20)], y_bmm_median[2][0:round:(round//20)], label="v="+str(v_mom_list[2]), color="green", linestyle="-", linewidth='3')
        plt.xlabel("Iteration", fontsize=22)
        plt.ylabel("Median Regret", fontsize=22)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend(fontsize=16)
        plt.grid(linestyle=":")
        plt.savefig("./figures/student_" + str(df).replace('.', '_') + "_noise_" + str(round) + "_bmm_median_uiUt.pdf")
        # plt.show()

        # SupBTC_mom

        # Mean
        plt.figure(figsize=(10, 6), dpi=100)
        plt.plot(x[0:round:(round//20)], y_btc_mean[0][0:round:(round//20)], label="v="+str(v_mom_list[0]), color="tab:cyan", linestyle="-", linewidth='3')
        plt.plot(x[0:round:(round//20)], y_btc_mean[1][0:round:(round//20)], label="v="+str(v_mom_list[1]), color="red", linestyle="-", linewidth='3')
        plt.plot(x[0:round:(round//20)], y_btc_mean[2][0:round:(round//20)], label="v="+str(v_mom_list[2]), color="green", linestyle="-", linewidth='3')
        plt.xlabel("Iteration", fontsize=22)
        plt.ylabel("Mean Regret", fontsize=22)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend(fontsize=16)
        plt.grid(linestyle=":")
        plt.savefig("./figures/student_" + str(df).replace('.', '_') + "_noise_" + str(round) + "_btc_mean_uiUt.pdf")
        # plt.show()

        # Median
        plt.figure(figsize=(10, 6), dpi=100)
        plt.plot(x[0:round:(round//20)], y_btc_median[0][0:round:(round//20)], label="v="+str(v_mom_list[0]), color="tab:cyan", linestyle="-", linewidth='3')
        plt.plot(x[0:round:(round//20)], y_btc_median[1][0:round:(round//20)], label="v="+str(v_mom_list[1]), color="red", linestyle="-", linewidth='3')
        plt.plot(x[0:round:(round//20)], y_btc_median[2][0:round:(round//20)], label="v="+str(v_mom_list[2]), color="green", linestyle="-", linewidth='3')
        plt.xlabel("Iteration", fontsize=22)
        plt.ylabel("Median Regret", fontsize=22)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend(fontsize=16)
        plt.grid(linestyle=":")
        plt.savefig("./figures/student_" + str(df).replace('.', '_') + "_noise_" + str(round) + "_btc_median_uiUt.pdf")
        # plt.show()
