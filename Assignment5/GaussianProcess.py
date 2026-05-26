from scipy.optimize import minimize
from matplotlib import pyplot as plt
import numpy as np
import math

# -------------------- LOAD DATA --------------------
def createMatrix(filename):
    data = [[0.0 for _ in range(2)] for _ in range(34)]
    with open(filename, "r") as file:
        for i in range(34):
            values = file.readline().split()
            data[i][0] = float(values[0])
            data[i][1] = float(values[1])
    return data
# -------------------- KERNEL --------------------
def rationalQuadratic(x1, x2, alpha, l, sigma):
    diff = x1 - x2
    sqdist = diff * diff
    base = 1.0 + (sqdist / (2.0 * alpha * l * l))
    return (sigma * sigma) * (base ** (-alpha))
# -------------------- BUILD KERNEL --------------------
def buildKernel(X, alpha, l, sigma, beta):
    n = len(X)
    K = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            K[i][j] = rationalQuadratic(X[i], X[j], alpha, l, sigma)
            if i == j:
                K[i][j] += 1.0 / beta
    return K
# -------------------- MATRIX OPS --------------------
def matVecMul(A, v):
    n = len(A)
    res = [0.0 for _ in range(n)]
    for i in range(n):
        for j in range(n):
            res[i] += A[i][j] * v[j]
    return res

def dot(a, b):
    s = 0.0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s
# -------------------- INVERSE --------------------
def invertMatrix(A):
    n = len(A)
    I = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        I[i][i] = 1.0
    for i in range(n):
        diag = A[i][i]
        for j in range(n):
            A[i][j] /= diag
            I[i][j] /= diag
        for k in range(n):
            if k == i:
                continue
            factor = A[k][i]
            for j in range(n):
                A[k][j] -= factor * A[i][j]
                I[k][j] -= factor * I[i][j]
    return I
# -------------------- DETERMINANT --------------------
def determinant(A):
    n = len(A)
    det = 1.0
    sign = 1

    for i in range(n):
        pivot = i
        for j in range(i + 1, n):
            if abs(A[j][i]) > abs(A[pivot][i]):
                pivot = j
        if abs(A[pivot][i]) < 1e-12:
            return 0
        if pivot != i:
            A[i], A[pivot] = A[pivot], A[i]
            sign *= -1
        for j in range(i + 1, n):
            factor = A[j][i] / A[i][i]
            for k in range(i, n):
                A[j][k] -= factor * A[i][k]
        det *= A[i][i]

    return sign * det
# -------------------- NLML --------------------
def computeNMLL(K, K_inv, Y):
    n = len(Y)
    detK = determinant([row[:] for row in K])
    logDet = math.log(abs(detK) + 1e-12)

    temp = matVecMul(K_inv, Y)
    quad = dot(Y, temp)
    constant = (n / 2.0) * math.log(2.0 * math.pi)
    return 0.5 * logDet + 0.5 * quad + constant
# -------------------- GP PREDICTION --------------------
def GPTestPointsPredict(K_inv, X, Y, N, alpha, l, sigma, beta, filename):
    start = -60
    end = 60
    step = (end - start) / (N - 1)

    with open(filename, "w") as out:
        temp = matVecMul(K_inv, Y)
        for t in range(N):
            x_star = start + t * step
            k = [0.0 for _ in range(len(X))]
            for i in range(len(X)):
                k[i] = rationalQuadratic(X[i], x_star, alpha, l, sigma)
            mean = dot(k, temp)
            temp2 = matVecMul(K_inv, k)
            var = (rationalQuadratic(x_star, x_star, alpha, l, sigma)- dot(k, temp2))
            std = math.sqrt(max(var, 1e-12))
            out.write(f"{x_star} {mean} {mean + 1.96 * std} {mean - 1.96 * std}\n")
# -------------------- OBJECTIVE (UPDATED) --------------------
def objective(theta, X, Y, beta):
    alpha = theta[0]
    l = theta[1]
    sigma = theta[2]

    K = buildKernel(X, alpha, l, sigma, beta)
    K_inv = invertMatrix([row[:] for row in K])

    return computeNMLL(K, K_inv, Y)
#----------------------Plotting-----------------
def plot_gp(before_file, train_file, output_image):
    # data loading
    train = np.loadtxt(train_file)
    X_train = train[:, 0]
    Y_train = train[:, 1]
    before = np.loadtxt(before_file)
    x = before[:, 0]
    mean = before[:, 1]
    upper = before[:, 2]
    lower = before[:, 3]
    # plotting
    plt.figure(figsize=(12, 8))
    # 95% confidence interval
    plt.fill_between(x, lower, upper, color="lightblue",label="95% Confidence Interval")
    # mean line
    plt.plot(x, mean,color="blue",linewidth=2,label="Mean")
    # training points
    plt.scatter(X_train, Y_train,color="black",s=60,label="Training Data")
    # titles legends
    plt.title("Gaussian Process Regression")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend()
    #saving
    plt.savefig(output_image, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", output_image)
# -------------------- MAIN --------------------
def main():
    data = createMatrix("./data/input.data")

    X = [0.0] * 34
    Y = [0.0] * 34

    for i in range(34):
        X[i] = data[i][0]
        Y[i] = data[i][1]

    beta = 5.0
    alpha = 1.0
    l = 1.0
    sigma = 1.0

    # ---------------- TRAIN FILE ----------------
    with open("./output/train.txt", "w") as train:
        for i in range(34):
            train.write(f"{X[i]} {Y[i]}\n")

    # ---------------- BEFORE OPT ----------------
    K_before = buildKernel(X, alpha, l, sigma, beta)
    Kinv_before = invertMatrix([row[:] for row in K_before])
    print("alpha =", alpha, "l =", l, "sigma =", sigma)
    GPTestPointsPredict(Kinv_before, X, Y, 200,alpha, l, sigma, beta,"./output/output_before.txt")
    print("Before NLML =", computeNMLL(K_before, Kinv_before, Y))
    # ---------------- OPTIMIZATION (SCIPY) ----------------
    result = minimize(objective,[alpha, l, sigma],args=(X, Y, beta),method='L-BFGS-B',bounds=[(1e-5, None), (1e-5, None), (1e-5, None)])
    alpha, l, sigma = result.x

    print("Optimized parameters:")
    print("alpha =", alpha, "l =", l, "sigma =", sigma)
    # ---------------- AFTER OPT ----------------
    K_after = buildKernel(X, alpha, l, sigma, beta)
    Kinv_after = invertMatrix([row[:] for row in K_after])

    GPTestPointsPredict(Kinv_after, X, Y, 200,alpha, l, sigma, beta,"./output/output_after.txt")
    print("After NLML =", computeNMLL(K_after, Kinv_after, Y))

    plot_gp("./output/output_before.txt","./output/train.txt","./output/gp_result.jpg")
    plot_gp("./output/output_after.txt","./output/train.txt","./output/gpOpt_result.jpg")

if __name__ == "__main__":
    main()