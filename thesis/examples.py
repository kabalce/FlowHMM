import numpy as np
from hmmlearn import hmm
import array_to_latex as a2l
from ssm.plots import gradient_cmap, white_to_color_cmap
import matplotlib.pyplot as plt
import seaborn as sns
import urllib
import scipy

A = np.array([[0.1, 0.9, 0], [0, 0.4, 0.6], [0.2, 0, 0.8]])

vals, vecs = np.linalg.eig(A.T)

vec1 = vecs[:, np.isclose(vals, 1)].reshape(-1)
stationary = vec1 / vec1.sum()

print([2/14,  3/14, 9/14])
print(stationary.real)

pi = np.array([2/14,  3/14, 9/14])

# sample example
np.random.seed(2023)

res = [np.random.choice([1, 2, 3], p=pi)]
for i in range(16):
    res.append(np.random.choice([1, 2, 3], p=A[res[i]-1]))

print(res)

colors_ = ['blue',  'teal', 'purple']

print(", ".join(["\\textcolor{" + colors_[r - 1] + "}{" + str(r) + "}" for r in res]))

# Example Discrete HMM

B = np.array([[0.5, 0.15, 0.15, 0.2],
              [0.2, 0.6, 0.1, 0.1],
              [0.1, 0.2, 0.2, 0.5]])

res2 = []
for i in range(16):
    res2.append(np.random.choice(['a', 'b', 'c',  'd'], p=B[res[i]-1]))

print(res2)

print(", ".join(["\\textcolor{" + colors_[c - 1] + "}{" + v + "}" for c,  v in zip(res, res2)]))

a2l.to_ltx(B, arraytype = 'array')

# Example Gaussian HMM

print("\n\nGaussian Parameters\n")

mu = np.array([[0.0, 0.0], [3.0, -3.0], [4.0, 3.0]])
Sigma = np.array([[[1, -.4], [-.4, .8]], [[.6, -.5], [-.5, 1.2]], [[.9, .6], [.6, 1.7]]])

a2l.to_ltx(mu[0], arraytype = 'array')
a2l.to_ltx(mu[1], arraytype = 'array')
a2l.to_ltx(mu[2], arraytype = 'array')

a2l.to_ltx(Sigma[0], arraytype = 'array')
a2l.to_ltx(Sigma[1], arraytype = 'array')
a2l.to_ltx(Sigma[2], arraytype = 'array')

x1, y1 = -3,  -7
x2, y2 = 7, 7

XX, YY = np.meshgrid(np.linspace(x1, x2, 100), np.linspace(y1, y2, 100))
data = np.column_stack((XX.ravel(), YY.ravel()))
lls = np.concatenate([scipy.stats.multivariate_normal(mu[i], Sigma[i]).pdf(data).reshape(-1, 1) for i in range(3)], axis=1)

with urllib.request.urlopen('https://xkcd.com/color/rgb.txt') as f:
    colors = f.readlines()
color_names = [str(c)[2:].split('\\t')[0] for c in colors[1:]]

color_names = colors_

colors = sns.xkcd_palette(color_names)
cmap = gradient_cmap(colors)

plt.figure(figsize=(5, 5))
for k in range(3):
    plt.contour(XX, YY, np.exp(lls[:, k]).reshape(XX.shape), cmap=white_to_color_cmap(colors[k]), levels=8)

res3 = []
for i in range(16):
    res3.append(np.random.multivariate_normal(mu[res[i]-1], Sigma[res[i]-1]))


plt.plot([r[0] for r in res3], [r[1] for r in res3],  linestyle='-', marker='o', lw=.3)
for i in range(16):
    x = np.random.uniform(-.2, .2) if i not in [7] else -.2
    y = .2 if i not in [14,  9, 13] else -.2
    plt.text(res3[i][0] + x, res3[i][1] + y, str(i+1), in_layout=True)

plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("Gaussian HMM Example")
plt.savefig('gaussian_hmm_example.eps', format='eps')
plt.show()
plt.close()


print(res3)

print(",\n ".join(["\\textcolor{" + colors_[c - 1] + "}{\\left(\\begin{array}{c} " + str("\\\\".join([str(w) for w in v.round(3)])) + "\\end{array}\\right)}" for c,  v in zip(res, res3)]))

# Example  co-occurrence

print("calculate Omega")
a2l.to_ltx(np.array([[1, 2, 1, 0], [0, 2, 3, 0],  [2, 2, 0, 1],  [0, 0, 1, 0]]) / 15, frmt = '{:6.4f}', arraytype = 'array')

print(pi.reshape(-1,  1) * A)

a2l.to_ltx((np.array([2,  3, 9]).reshape(-1,  1) * A * 10).astype(int), arraytype = 'array')

a2l.to_ltx(np.array([2,  3, 9]).reshape(-1,  1) * A / 14, frmt = '{:6.4f}', arraytype = 'array')

S_140 = (np.array([2,  3, 9]).reshape(-1,  1) * A).astype(int)
B_10 = B * 10

print(S_140.shape)
print(B_10.shape)
print((S_140 @ B_10).shape)
a2l.to_ltx(B_10.T @ (S_140 @ B_10), frmt = '{:6.1f}', arraytype = 'array')


a2l.to_ltx(B_10.T @ (S_140 @ B_10) / 14000, frmt = '{:6.5f}', arraytype = 'array')