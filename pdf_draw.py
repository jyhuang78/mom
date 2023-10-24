import numpy as np
from scipy.stats import t
from scipy.stats import norm
import matplotlib.pyplot as plt

distribs = [t(0.5), t(1), t(3), norm]
distrib_names = ['t(0.5)', 't(1)', 't(3)', 'norm']
# for distrib, distrib_name in zip(distribs, distrib_names):
#     alpha = np.log(4) / np.log(distrib.isf(1/8))
#     print(distrib_name + ": alpha = " + str(alpha))

x = np.linspace(-10, 10, 1000)
plt.figure()
plt.plot(x, distribs[0].pdf(x), color='r', label=distrib_names[0])
plt.plot(x, distribs[1].pdf(x), color='g', label=distrib_names[1])
plt.plot(x, distribs[2].pdf(x), color='gray', label=distrib_names[2])
plt.plot(x, distribs[3].pdf(x), color='blue', label=distrib_names[3])
plt.ylabel("Density", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.savefig("./figures/pdf-super-heavy-tailed.pdf")
plt.show()
