import numpy as np
from scipy.stats import t
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


def supbtc(round, d, K, theta, epsilon, v):
    t = 0
    historical_information = np.zeros((round, d))
    historical_payoff = np.zeros((round, 1))
    cul_regret = 0

    with open("./data/supbtc_" + str(round) + "_student_" + str(df) + "_noise.txt", "w") as f:
        while t < round:
            print("round", t)
            cur_contexts, best_arm = get_contexts(K, d)
            estimated_payoff, width = btc(t, historical_information[:t,:], historical_payoff[:t,:], cur_contexts, d, K, epsilon, v)
            selected_arm = np.argmax(estimated_payoff + width)        
            selected_context = cur_contexts[selected_arm]
            cur_payoff = get_payoff(selected_context, theta)
            historical_payoff[t] = cur_payoff
            historical_information[t] = selected_context
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

    with open("./data/supbtc_mom_" + str(round_total) + "_student_" + str(df) + "_noise_vareps_" + str(varepsilon) + "_path_" + str(i_path) + ".txt", "w") as f:
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
    num_path = 10

    # df = 3
    # # varepsilon = 0.5
    # # n_tilde = (16 * np.log(2 * round / delta))**(1/varepsilon)
    # n_tilde_btc = 256
    # # heavy-tailed
    # epsilon = (df - 1) / 2
    # epsilon_mom = 1
    # v = t(df).expect(lambda x: abs(x) ** (1 + epsilon))
    # v_mom_btc = 1/2
    
    # round, K, d = 100000, 20, 10
    # theta = np.ones(shape=(d, 1)) / np.sqrt(d)

    # # supbtc(round, d, K, theta, epsilon, v)
    # for i_path in range(num_path):
    #     supbtc_mom(round, d, K, theta, epsilon_mom, v_mom_btc, 0.5, n_tilde_btc)
    #     supbtc_mom(round, d, K, theta, epsilon_mom, v_mom_btc, 1, n_tilde_btc)
    
    # for varepsilon in [0.5, 1]:
    #     y = np.zeros((num_path, round))
    #     for i_path in range(num_path):
    #         with open("./data/supbtc_mom_" + str(round) + "_student_" + str(df) + "_noise_vareps_" + str(varepsilon) + "_path_" + str(i_path) + ".txt", "r") as f:
    #             y[i_path] = list(map(float, f.readline().strip().split()))[:round]
    #     y_many_paths = np.mean(y, axis=0)
    #     np.savetxt("./data/supbtc_mom_" + str(round) + "_student_" + str(df) + "_noise_vareps_" + str(varepsilon) + ".txt", y_many_paths, fmt='%.15f', newline='\t')


    # # draw
    # x = np.arange(0, round)
    # plt.figure(figsize=(10, 6), dpi=100)


    # # with open("./data/supbtc_" + str(round) + "_student_" + str(df) + "_noise.txt", "r") as f:
    # #     y_btc = list(map(float, f.readline().strip().split()))
    # # plt.plot(x[0:round:(round//20)], y_btc[0:round:(round//20)], label="SupBTC", color="tab:cyan", linestyle="-", linewidth='3')

    # with open("./data/supbtc_mom_" + str(round) + "_student_" + str(df) + "_noise_vareps_0.5.txt", "r") as f:
    #     y_btc = list(map(float, f.readline().strip().split()))
    # plt.plot(x[0:round:(round//20)], y_btc[0:round:(round//20)], label="SupBTC_mom, varepsilon=0.5", color="green", linestyle="-", linewidth='3')

    # with open("./data/supbtc_mom_" + str(round) + "_student_" + str(df) + "_noise_vareps_1.txt", "r") as f:
    #     y_btc = list(map(float, f.readline().strip().split()))
    # plt.plot(x[0:round:(round//20)], y_btc[0:round:(round//20)], label="SupBTC_mom, varepsilon=1", color="red", linestyle="-", linewidth='3')

    # plt.xlabel("Iteration", fontsize=22)
    # plt.ylabel("Regret", fontsize=22)
    # plt.xticks(fontsize=18)
    # plt.yticks(fontsize=18)
    # plt.legend(fontsize=16)
    # plt.grid(linestyle=":")
    # plt.savefig("./figures/student_" + str(df).replace('.', '_') + "_noise_" + str(round) + "_mom_vs_median.pdf")
    # plt.show()



    df = 0.5
    # varepsilon = 0.5
    # n_tilde = (16 * np.log(2 * round / delta))**(1/varepsilon)
    n_tilde_btc = 400
    # heavy-tailed
    epsilon = 0.01
    epsilon_mom = 1
    v = 10000
    v_mom_btc = 1
    
    round, K, d = 100000, 20, 10
    theta = np.ones(shape=(d, 1)) / np.sqrt(d)

    # supbtc(round, d, K, theta, epsilon, v)
    for i_path in range(num_path):
        supbtc_mom(round, d, K, theta, epsilon_mom, v_mom_btc, 0.5, n_tilde_btc)
        supbtc_mom(round, d, K, theta, epsilon_mom, v_mom_btc, 1, n_tilde_btc)
    
    for varepsilon in [0.5, 1]:
        y = np.zeros((num_path, round))
        for i_path in range(num_path):
            with open("./data/supbtc_mom_" + str(round) + "_student_" + str(df) + "_noise_vareps_" + str(varepsilon) + "_path_" + str(i_path) + ".txt", "r") as f:
                y[i_path] = list(map(float, f.readline().strip().split()))[:round]
        y_many_paths = np.mean(y, axis=0)
        np.savetxt("./data/supbtc_mom_" + str(round) + "_student_" + str(df) + "_noise_vareps_" + str(varepsilon) + ".txt", y_many_paths, fmt='%.15f', newline='\t')


    # draw
    x = np.arange(0, round)
    plt.figure(figsize=(10, 6), dpi=100)


    # with open("./data/supbtc_" + str(round) + "_student_" + str(df) + "_noise.txt", "r") as f:
    #     y_btc = list(map(float, f.readline().strip().split()))
    # plt.plot(x[0:round:(round//20)], y_btc[0:round:(round//20)], label="SupBTC", color="tab:cyan", linestyle="-", linewidth='3')

    with open("./data/supbtc_mom_" + str(round) + "_student_" + str(df) + "_noise_vareps_0.5.txt", "r") as f:
        y_btc = list(map(float, f.readline().strip().split()))
    plt.plot(x[0:round:(round//20)], y_btc[0:round:(round//20)], label="SupBTC_mom, varepsilon=0.5", color="green", linestyle="-", linewidth='3')

    with open("./data/supbtc_mom_" + str(round) + "_student_" + str(df) + "_noise_vareps_1.txt", "r") as f:
        y_btc = list(map(float, f.readline().strip().split()))
    plt.plot(x[0:round:(round//20)], y_btc[0:round:(round//20)], label="SupBTC_mom, varepsilon=1", color="red", linestyle="-", linewidth='3')

    plt.xlabel("Iteration", fontsize=22)
    plt.ylabel("Regret", fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=16)
    plt.grid(linestyle=":")
    plt.savefig("./figures/student_" + str(df).replace('.', '_') + "_noise_" + str(round) + "_mom_vs_median.pdf")
    plt.show()
