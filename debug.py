# min_max = {
#     "thickness": [0.87598526, 6.255515],
#     "intensity": [66.601204, 254.90317],
# }
#
# # [-1, 1] -> [min, max]
# unormalize = {
#     "thickness": lambda x: (x + 1) / 2 * (min_max["thickness"][1] - min_max["thickness"][0]) + min_max["thickness"][0],
#     "intensity": lambda x: (x + 1) / 2 * (min_max["intensity"][1] - min_max["intensity"][0]) + min_max["intensity"][0],
# }
#
# t = -0.3981
# i = -0.4127
#
# t = unormalize["thickness"](t)
# i = unormalize["intensity"](i)
#
# # 小数点后2位
# print(f"t = {t:.2f} \ni = {i:.2f}")

import numpy as np
import matplotlib.pyplot as plt

num_timesteps = 1000
timesteps = 50

skip = num_timesteps // timesteps
seq1 = range(0, num_timesteps, skip)
seq2 = (
        np.linspace(
            0, np.sqrt(num_timesteps * 0.8), timesteps
        )
        ** 2
)
seq2 = [int(s) for s in list(seq2)]

# [0, seq1[0]], [1, seq1[1]], [2, seq1[2]], ...
# [0, seq2[0]], [1, seq2[1]], [2, seq2[2]], ...

plt.plot(seq1, label="uniform")
plt.plot(seq2, label="quadratic")
plt.xlabel("Step")
plt.ylabel("T")
plt.legend()
plt.show()