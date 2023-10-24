import numpy as np
import matplotlib.pyplot as plt

delta = 0.01
T = 10000

alpha = np.linspace(0, 3, 100)
vareps = (np.log(16) + np.log(np.log(T/delta))) / (np.log(16) + np.log(np.log(T/delta)) + 2/alpha * np.log(4) + np.log(np.log(4/delta)))

plt.figure()

plt.plot(alpha, vareps)
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\varepsilon$')
plt.grid(linestyle=":")
plt.savefig("./figures/vareps-vs-alpha.pdf")
plt.show()
