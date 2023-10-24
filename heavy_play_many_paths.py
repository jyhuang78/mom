import numpy as np
from scipy.stats import t


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
    with open("./data/mom_" + str(round) + "_student_" + str(df) + "_noise_path_" + str(i_path) + ".txt", "w") as f:
        while t < m:
            # print("round", t)
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
def crt(round, d, K, theta, epsilon, v):
    a, inverse_a = np.eye(d), np.eye(d)
    b = np.zeros((d, 1))
    hat_theta = np.zeros((d, 1))
    beta, t = 1, 0
    cul_regret = 0
    with open("./data/crt_" + str(round) + "_student_" + str(df) + "_noise_path_" + str(i_path) + ".txt", "w") as f:
        while t < round:
            # print("round", t)
            cur_contexts, best_arm = get_contexts(K, d)
            # print(np.linalg.norm(hat_theta - theta))
            estimated_payoff = np.dot(cur_contexts, hat_theta)  # K*1
            width = []
            for i in range(K):
                width.append(beta * np.sqrt(np.dot(np.dot(cur_contexts[i], inverse_a), cur_contexts[i].T)))
            width = np.array(width).reshape(K, 1)
            estimated_payoff += width
            selected_arm = int(np.argmax(estimated_payoff))
            cul_regret += np.dot(cur_contexts[best_arm] - cur_contexts[selected_arm], theta)
            f.write(str(cul_regret[0]) + '\t')
            t += 1
            cur_payoff = get_payoff(cur_contexts[selected_arm], theta)  # 1*1
            cur_payoff = 0 if abs(cur_payoff[0]) > t ** (1 / (2 * (1 + epsilon))) else cur_payoff[0]
            b += cur_contexts[selected_arm].reshape(d, 1) * cur_payoff
            a += np.dot(cur_contexts[selected_arm].reshape(d, 1), cur_contexts[selected_arm].reshape(1, d))
            inverse_a = np.linalg.inv(a)
            hat_theta = np.dot(inverse_a, b)
            beta = 2 * np.sqrt(2 * t ** (1 / (1 + epsilon)) * np.log(np.linalg.det(a) ** (1 / 2) / delta)) + np.sqrt(
                2 * v * v * np.log(t)) + 1
            beta /= 10


def get_estimator(hat_theta, a, r):
    dis = [[0] * r for _ in range(r)]
    for i in range(r):
        for j in range(i + 1, r):
            dis[i][j] = np.dot(np.dot(hat_theta[:, i].reshape(1, d) - hat_theta[:, j].reshape(1, d), a),
                               hat_theta[:, i].reshape(1, d).T - hat_theta[:, j].reshape(1, d).T)
            dis[j][i] = dis[i][j]
    dis = np.array(dis)
    median = np.median(dis, axis=1, keepdims=True)
    index = int(np.argmin(median))
    return hat_theta[:, index]


# Global: delta
# Implicit: theta = np.ones(shape=(d, 1)) / np.sqrt(d)
def menu(round, d, K, theta, epsilon, v):
    # r = int(24 + 24 * np.log(round / delta))
    r = 51
    m = int(round / r) + 1
    a, inverse_a = np.eye(d), np.eye(d)
    b = np.zeros((d, r))  # d*r
    hat_theta = np.zeros((d, 1))
    beta = 1
    t = 0

    cul_regret, res = 0, []
    with open("./data/menu_" + str(round) + "_student_" + str(df) + "_noise_path_" + str(i_path) + ".txt", "w") as f:
        while t < m:
            # print("round", t)
            cur_contexts, best_arm = get_contexts(K, d)
            # print(np.linalg.norm(hat_theta - theta))
            estimated_payoff = np.dot(cur_contexts, hat_theta)  # K*1
            width = []
            for i in range(K):
                width.append(beta * np.sqrt(np.dot(np.dot(cur_contexts[i], inverse_a), cur_contexts[i].T)))
            width = np.array(width).reshape(K, 1)
            # print(estimated_payoff.shape)
            estimated_payoff += width
            selected_arm = int(np.argmax(estimated_payoff))
            for _ in range(r):
                cul_regret += np.dot(cur_contexts[best_arm] - cur_contexts[selected_arm], theta)
                f.write(str(cul_regret[0]) + '\t')
            t += 1
            cur_payoff = get_payoff(cur_contexts[selected_arm], theta, r)  # 1*r
            for i in range(r):
                b[:, i] += cur_contexts[selected_arm] * cur_payoff[0][i]
            a += np.dot(cur_contexts[selected_arm].reshape(d, 1), cur_contexts[selected_arm].reshape(1, d))
            inverse_a = np.linalg.inv(a)
            estimators = np.dot(inverse_a, b)  # d*r
            hat_theta = get_estimator(estimators, a, r).reshape(d, 1)
            beta = 3 * (9 * d * v) ** (1 / (1 + epsilon)) * t ** ((1 - epsilon) / (2 * (1 + epsilon))) + 3
            beta /= 50


# Global: delta
def tofu(round, d, K, theta, epsilon, v):
    a, inverse_a = np.eye(d), np.eye(d)
    b = np.zeros((d, 1))  # d*1
    hat_theta = np.zeros((d, 1))
    historical_contexts = np.empty((round, d))
    historical_payoff = np.empty((round, 1))
    hat_y = np.empty((d, round))

    beta = 1
    t = 0

    cul_regret = 0
    with open("./data/tofu_" + str(round) + "_student_" + str(df) + "_noise_path_" + str(i_path) + ".txt", "w") as f:
        while t < round:
            print("round", t)
            trun = (v / np.log(2 * round / delta)) ** (1 / 2) * t ** ((1 - epsilon) / (2 * (1 + epsilon)))
            cur_contexts, best_arm = get_contexts(K, d)
            # print(np.linalg.norm(hat_theta - theta))
            estimated_payoff = np.dot(cur_contexts, hat_theta)  # K*1
            width = []
            for i in range(K):
                width.append(beta * np.sqrt(np.dot(np.dot(cur_contexts[i], inverse_a), cur_contexts[i].T)))
            width = np.array(width).reshape(K, 1)
            estimated_payoff += width
            selected_arm = int(np.argmax(estimated_payoff))
            cur_payoff = get_payoff(cur_contexts[selected_arm], theta)  # 1*1
            historical_contexts[t] = cur_contexts[selected_arm]
            historical_payoff[t] = cur_payoff[0]
            cul_regret += np.dot(cur_contexts[best_arm] - cur_contexts[selected_arm], theta)
            f.write(str(cul_regret[0]) + '\t')
            t += 1
            a += np.dot(cur_contexts[selected_arm].reshape(d, 1), cur_contexts[selected_arm].reshape(1, d))
            u, sigma, z = np.linalg.svd(a)
            half_a = np.dot(np.dot(u, np.diag(sigma**(-0.5))), z)  # d*d
            mid = np.dot(half_a, historical_contexts[:t, :].T)  # d*t
            inverse_a = np.linalg.inv(a)
            for i in range(d):
                for j in range(t):
                    hat_y[i][j] = historical_payoff[j][0] if abs(historical_payoff[j][0] * mid[i][j]) <= trun else 0
                b[i] = np.dot(mid[i], hat_y[i, :t].T)
            hat_theta = np.dot(half_a, b)  # d*1
            beta = 4 * np.sqrt(d) * v ** (1 / (1 + epsilon)) * (np.log(2 * d * round / delta)) ** (epsilon / (1 + epsilon)) * t ** ((1 - epsilon) / (2 * (1 + epsilon))) + 1
            beta /= 5


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

    with open("./data/supbmm_" + str(round) + "_student_" + str(df) + "_noise_path_" + str(i_path) + ".txt", "w") as f:
        while t < cur_round:
            # print("round", t)
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

    with open("./data/supbtc_" + str(round) + "_student_" + str(df) + "_noise_path_" + str(i_path) + ".txt", "w") as f:
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

    with open("./data/supbmm_mom_" + str(round_total) + "_student_" + str(df) + "_noise_path_" + str(i_path) + ".txt", "w") as f:
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

    with open("./data/supbtc_mom_" + str(round_total) + "_student_" + str(df) + "_noise_path_" + str(i_path) + ".txt", "w") as f:
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
    ### MODIFY BEFORE UPLOAD
    # algorithms = ['mom', 'crt', 'menu', 'tofu', 'supbmm', 'supbtc', 'supbmm_mom', 'supbtc_mom']
    algorithms = ['supbmm_mom', 'supbtc_mom']
    ### MODIFY BEFORE UPLOAD

    for df in [3, 1.02]:
        
        varepsilon = 0.5
        # n_tilde = (16 * np.log(2 * round / delta))**(1/varepsilon)
        n_tilde_bmm = 64
        n_tilde_btc = 256
        # heavy-tailed
        epsilon = (df - 1) / 2
        epsilon_mom = 1
        v = t(df).expect(lambda x: abs(x) ** (1 + epsilon))
        v_mom_bmm = 1
        v_mom_btc = 1 / 2
        
        round, K, d = 10000, 20, 10
        theta = np.ones(shape=(d, 1)) / np.sqrt(d)
        
        for i_path in range(num_path):
            # mom(round, d, K, theta, epsilon, v)
            # crt(round, d, K, theta, epsilon, v)
            # menu(round, d, K, theta, epsilon, v)
            # tofu(round, d, K, theta, epsilon, v)
            # supbmm(round, d, K, theta, epsilon, v)
            # supbtc(round, d, K, theta, epsilon, v)
            supbmm_mom(round, d, K, theta, epsilon_mom, v_mom_bmm, varepsilon, n_tilde_bmm)
            supbtc_mom(round, d, K, theta, epsilon_mom, v_mom_btc, varepsilon, n_tilde_btc)
        
    for df in [3, 1.02]:
        for algorithm in algorithms:
            y = np.zeros((num_path, round))
            for i_path in range(num_path):
                with open("./data/" + algorithm + "_" + str(round) + "_student_" + str(df) + "_noise_path_" + str(i_path) + ".txt", "r") as f:
                    y[i_path] = list(map(float, f.readline().strip().split()))[:round]
            y_many_paths = np.mean(y, axis=0)
            np.savetxt("./data/" + algorithm + "_" + str(round) + "_student_" + str(df) + "_noise.txt", y_many_paths, fmt='%.15f', newline='\t')
