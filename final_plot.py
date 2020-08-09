import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import spline
#x_index 13:25
T = np.arange(13*0.04, 26*0.04, 0.04)
pr = [0.91666235, 0.90597066, 0.90885995, 0.91446838, 0.9200237, 0.92295502, 0.91495791, 0.91616153, 0.91978087, 0.92462974, 0.92511059, 0.92523344, 0.89024176]
re = [0.87131341, 0.88541667, 0.89060386, 0.87542572, 0.89402476, 0.90565217, 0.91033514, 0.89950181, 0.92297705, 0.90228865, 0.88233092, 0.88608998, 0.90612319]
f1 = [0.8933155, 0.89509849, 0.89945284, 0.89418533, 0.90647689, 0.91406733, 0.91247852, 0.90763855, 0.9211483, 0.91313116, 0.90296348, 0.90510605, 0.89796513]
T_new = np.linspace(T.min(), T.max(), 300)
pr_smooth = spline(T, pr, T_new)
re_smooth = spline(T, re, T_new)
f1_smooth = spline(T, f1, T_new)
plt.plot(T_new, pr_smooth, 'r', label = '$P_{r}$')
plt.plot(T_new, re_smooth, 'g', label = '$R_{e}$')
plt.plot(T_new, f1_smooth, 'b', label = '$F1 score$')
plt.ylim(0.8, 1)
plt.xlabel('perception and cognition time/seconds')
plt.ylabel('percentage')
plt.legend()
plt.show()
