import numpy as np

x = np.arange(0, 8, 1)
y = np.arange(0, 4, 1)

X, Y = np.meshgrid(x, y)

print(X)
# algorytm Thomasa
def solve_vectorized(a, b, c, d):
    n = d.shape[0]

    # zapisujemy same przekątne bo poza nimi są zera
    ac = a[:, np.newaxis]  # dolna przekątma
    bc = b[:, np.newaxis].copy()  # glowna przekątna
    cc = c[:, np.newaxis]  # gorna przekątna
    dc = d  # prawa strona rownania

    # eliminacja
    for i in range(1, n):
        m = ac[i - 1] / bc[i - 1]
        bc[i] = bc[i] - m * cc[i - 1]
        dc[i] = dc[i] - m * dc[i - 1]

    # podstawianie wsteczne
    x = np.zeros_like(dc)
    x[-1] = dc[-1] / bc[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (dc[i] - cc[i] * x[i + 1]) / bc[i]
    return x
