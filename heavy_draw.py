import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt

if __name__ == "__main__":
    for df in [3, 1.02]:

        round = 10000

        # draw
        x = np.arange(0, round)
        plt.figure(figsize=(10, 6), dpi=100)

        with open("./data/mom_" + str(round) + "_student_" + str(df) + "_noise.txt", "r") as f:
            y_mom = list(map(float, f.readline().strip().split()))
        plt.plot(x[0:round:(round//20)], y_mom[0:round:(round//20)], label="MoM", color="tab:blue", linestyle="-", linewidth='3')

        with open("./data/crt_" + str(round) + "_student_" + str(df) + "_noise.txt", "r") as f:
            y_crt = list(map(float, f.readline().strip().split()))
        plt.plot(x[0:round:(round//20)], y_crt[0:round:(round//20)], label="CRT", color="tab:orange", linestyle="-", linewidth='3')

        with open("./data/menu_" + str(round) + "_student_" + str(df) + "_noise.txt", "r") as f:
            y_menu = list(map(float, f.readline().strip().split()))
        plt.plot(x[0:round:(round//20)], y_menu[0:round:(round//20)], label="MENU", color="tab:brown", linestyle="-", linewidth='3')

        with open("./data/tofu_" + str(round) + "_student_" + str(df) + "_noise.txt", "r") as f:
            y_tofu = list(map(float, f.readline().strip().split()))
        plt.plot(x[0:round:(round//20)], y_tofu[0:round:(round//20)], label="TOFU", color="springgreen", linestyle="-", linewidth='3')

        with open("./data/supbmm_" + str(round) + "_student_" + str(df) + "_noise.txt", "r") as f:
            y_bmm = list(map(float, f.readline().strip().split()))
        plt.plot(x[0:round:(round//20)], y_bmm[0:round:(round//20)], label="SupBMM", color="tab:olive", linestyle="-", linewidth='3')

        with open("./data/supbtc_" + str(round) + "_student_" + str(df) + "_noise.txt", "r") as f:
            y_btc = list(map(float, f.readline().strip().split()))
        plt.plot(x[0:round:(round//20)], y_btc[0:round:(round//20)], label="SupBTC", color="tab:cyan", linestyle="-", linewidth='3')

        with open("./data/supbmm_mom_" + str(round) + "_student_" + str(df) + "_noise.txt", "r") as f:
            y_bmm = list(map(float, f.readline().strip().split()))
        plt.plot(x[0:round:(round//20)], y_bmm[0:round:(round//20)], label="SupBMM_mom", color="red", linestyle="-", linewidth='3')

        with open("./data/supbtc_mom_" + str(round) + "_student_" + str(df) + "_noise.txt", "r") as f:
            y_btc = list(map(float, f.readline().strip().split()))
        plt.plot(x[0:round:(round//20)], y_btc[0:round:(round//20)], label="SupBTC_mom", color="green", linestyle="-", linewidth='3')

        plt.xlabel("Iteration", fontsize=22)
        plt.ylabel("Regret", fontsize=22)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend(fontsize=16)
        plt.grid(linestyle=":")
        plt.savefig("./figures/student_" + str(df).replace('.', '_') + "_noise_" + str(round) + ".pdf")
        plt.show()
