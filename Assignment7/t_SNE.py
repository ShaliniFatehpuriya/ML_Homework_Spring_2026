import numpy as np
import pylab
# H(beta) for perplexity search
def Hbeta(D=np.array([]), beta=1.0):
    P = np.exp(-D.copy() * beta)
    sumP = np.sum(P)
    # prevent division by zero
    if sumP < 1e-12:
        sumP = 1e-12
    P = P / sumP
    H = np.log(sumP) + beta * np.sum(D * P)
    return H, P
# Convert X -> P (high dimensional similarities)
def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    for i in range(n):
        if i % 500 == 0:
            print(f"Computing P-values for point {i} of {n}...")
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] *= 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] /= 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP
    print("Mean sigma:", np.mean(np.sqrt(1 / beta)))
    return P

# PCA 
def pca(X=np.array([]), no_dims=50):

    print("Preprocessing using PCA...")
    X = X - np.mean(X, axis=0)
    cov = np.dot(X.T, X)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    Y = np.dot(X, eigvecs[:, :no_dims])
    return Y
# MAIN FUNCTION: supports BOTH t-SNE and Symmetric SNE
def tsne(X=np.array([]),no_dims=2,initial_dims=50,perplexity=30.0,method="tsne"):
    if method not in ["tsne", "sne"]:
        raise ValueError("method must be 'tsne' or 'sne'")
    X = pca(X, initial_dims).real
    (n, d) = X.shape
    max_iter = 1000
    eta = 500
    initial_momentum = 0.5
    final_momentum = 0.8
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))
    history=[]

    P = x2p(X, 1e-5, perplexity)
    P = P + P.T
    P = P / np.sum(P)
    P = P * 4.0
    P = np.maximum(P, 1e-12)

    Q_final = None

    for iter in range(max_iter):
        sum_Y = np.sum(np.square(Y), 1)
        dist = np.add(np.add(-2. * np.dot(Y, Y.T), sum_Y).T,sum_Y)

        if method == "tsne":
            num = 1. / (1. + dist)     # Student-t
        else:
            num = np.exp(-dist)        # Gaussian (SNE)
        num[range(n), range(n)] = 0.
        Q = (num + num.T) / (2. * np.sum(num))
        Q = np.maximum(Q, 1e-12)
        Q_final = Q
        PQ = P - Q
        if method == "tsne":
            for i in range(n):
                dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i],(no_dims, 1)).T* (Y[i, :] - Y), 0)
        else:
            for i in range(n):
                dY[i, :] = 4 * np.sum(np.tile(PQ[:, i],(no_dims, 1)).T* (Y[i, :] - Y),0)
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum

        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y += iY
        Y -= np.mean(Y, axis=0)
        if iter % 10 == 0:
            history.append(Y.copy())

        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log((P + 1e-12) / (Q + 1e-12)))
            print(f"[{method.upper()}] Iter {iter+1}: cost {C}")
        if iter == 100:
            P /= 4.
    return Y, P, Q_final,history

if __name__ == "__main__":
    print("Loading MNIST...")
    X = np.loadtxt("./MNSIT database/mnist2500_X.txt")
    labels = np.loadtxt("./MNSIT database/mnist2500_labels.txt")
    perplexity = [15,20,50,65]
    for prep in perplexity:
        Y_tsne, P_tsne, Q_tsne,history_tsne = tsne(X,no_dims=2,initial_dims=50,perplexity=prep,method="tsne")
        Y_sne, P_sne, Q_sne,history_sne = tsne(X,no_dims=2,initial_dims=50,perplexity=prep,method="sne")
        np.save(f"./npy_files/history_tsne{prep}.npy", np.array(history_tsne, dtype=object), allow_pickle=True)
        np.save(f"./npy_files/history_sne{prep}.npy", np.array(history_sne, dtype=object), allow_pickle=True)
        np.save(f"./npy_files/P_tsne{prep}.npy", P_tsne)
        np.save(f"./npy_files/Q_tsne{prep}.npy", Q_tsne)
        np.save(f"./npy_files/P_sne{prep}.npy", P_sne)
        np.save(f"./npy_files/Q_sne{prep}.npy", Q_sne)
    pylab.figure(figsize=(12, 5))
    pylab.subplot(1, 2, 1)
    pylab.scatter(Y_tsne[:, 0], Y_tsne[:, 1], 20, labels)
    pylab.title("t-SNE")
    pylab.subplot(1, 2, 2)
    pylab.scatter(Y_sne[:, 0], Y_sne[:, 1], 20, labels)
    pylab.title("Symmetric SNE")
    pylab.show()