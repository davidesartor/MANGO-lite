# %%
import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(0.99, 0, 100)
r = -np.log2(1 - t) / 2
plt.plot(t, r)
plt.plot(t, t)
plt.ylim(0, 1)
plt.show()
