import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter


# DATA LOADING
def CreateMatrix(path):
    path = Path(path)
    X, Y = [], []

    for file in sorted(path.iterdir()):
        if file.is_file():
            img = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (50, 50))
            X.append(img.flatten())

            label = int(file.stem.split(".")[0].replace("subject", ""))
            Y.append(label)

    return np.array(X, dtype=np.float64), np.array(Y)


# KNN Class
class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        preds = []
        for x in X:
            d = np.linalg.norm(self.X - x, axis=1)
            idx = np.argsort(d)[:self.k]
            preds.append(Counter(self.y[idx]).most_common(1)[0][0])
        return np.array(preds)


# PCA CLASS
class PCA:
    def __init__(self, n_components=100):
        self.n_components = n_components

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        Xc = X - self.mean
        cov = np.dot(Xc, Xc.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        eigenfaces = np.dot(Xc.T, eigvecs)
        eigenfaces = eigenfaces / (np.linalg.norm(eigenfaces, axis=0, keepdims=True) + 1e-8)

        max_comp = min(self.n_components, eigenfaces.shape[1])
        self.components = eigenfaces[:, :max_comp]

    def transform(self, X):
        return np.dot(X - self.mean, self.components)

    def reconstruct(self, X):
        X2 = np.atleast_2d(X)
        rec = np.dot(self.transform(X2), self.components.T) + self.mean
        return rec[0] if X.ndim == 1 else rec


# LDA CLASS
class LDA:
    def __init__(self, n_components=10):
        self.n_components = n_components

    def fit(self, X, y):
        self.mean = np.mean(X, axis=0)
        classes = np.unique(y)

        Sw = np.zeros((X.shape[1], X.shape[1]))
        Sb = np.zeros((X.shape[1], X.shape[1]))

        for c in classes:
            Xc = X[y == c]
            mean_c = np.mean(Xc, axis=0)
            Sw += (Xc - mean_c).T @ (Xc - mean_c)

            n = Xc.shape[0]
            diff = (mean_c - self.mean).reshape(-1, 1)
            Sb += n * diff @ diff.T

        M = np.linalg.pinv(Sw + 1e-8 * np.eye(Sw.shape[0])) @ Sb
        eigvals, eigvecs = np.linalg.eig(M)
        idx = np.argsort(np.real(eigvals))[::-1]

        max_comp = min(self.n_components, len(classes) - 1, X.shape[1])
        self.components = np.real(eigvecs[:, idx[:max_comp]])

    def transform(self, X):
        return np.dot(X - self.mean, self.components)


# Kernel functions
def linear_kernel(X, Y):
    return X @ Y.T


def rbf_kernel(X, Y, gamma=1e-7):
    X2 = np.sum(X**2, axis=1).reshape(-1, 1)
    Y2 = np.sum(Y**2, axis=1).reshape(1, -1)
    return np.exp(-gamma * (X2 + Y2 - 2 * X @ Y.T))


def center_kernel(K):
    n = K.shape[0]
    one = np.ones((n, n)) / n
    return K - one @ K - K @ one + one @ K @ one


# Kernel PCA
class KernelPCA:
    def __init__(self, kernel="rbf", n_components=25):
        self.kernel = kernel
        self.n_components = n_components

    def _kernel(self, X, Y):
        if self.kernel == "rbf":
            return rbf_kernel(X, Y)
        elif self.kernel == "linear":
            return linear_kernel(X, Y)
        else:
            raise ValueError("Unknown kernel")

    def fit(self, X):
        self.X = X
        K = self._kernel(X, X)
        Kc = center_kernel(K)

        eigvals, eigvecs = np.linalg.eigh(Kc)
        idx = np.argsort(eigvals)[::-1]
        eigvals = np.real(eigvals[idx])
        eigvecs = np.real(eigvecs[:, idx])

        max_comp = min(self.n_components, X.shape[0])
        self.lambdas = eigvals[:max_comp]
        self.alphas = eigvecs[:, :max_comp]

        for i in range(self.alphas.shape[1]):
            self.alphas[:, i] /= (np.sqrt(self.lambdas[i]) + 1e-8)

        self.train_kernel = K
        self.train_kernel_centered = Kc
        self.K_row_mean = np.mean(K, axis=0)
        self.K_all_mean = np.mean(K)

    def transform(self, X):
        K = self._kernel(X, self.X)
        K_mean_rows = np.mean(K, axis=1, keepdims=True)
        Kc = K - self.K_row_mean.reshape(1, -1) - K_mean_rows + self.K_all_mean
        return Kc @ self.alphas


# Kernel LDA
class KernelLDA:
    def __init__(self, kernel="rbf", n_components=10):
        self.kernel = kernel
        self.n_components = n_components

    def _kernel(self, X, Y):
        if self.kernel == "rbf":
            return rbf_kernel(X, Y)
        elif self.kernel == "linear":
            return linear_kernel(X, Y)
        else:
            raise ValueError("Unknown kernel")

    def fit(self, X, y):
        self.X = X
        K = self._kernel(X, X)
        K = center_kernel(K)

        classes = np.unique(y)
        n = K.shape[0]
        M = np.mean(K, axis=0).reshape(-1, 1)

        Sw = np.zeros((n, n))
        Sb = np.zeros((n, n))

        for c in classes:
            idx = np.where(y == c)[0]
            Kc = K[:, idx]
            mc = np.mean(Kc, axis=1).reshape(-1, 1)
            Sw += (Kc - mc) @ (Kc - mc).T
            diff = mc - M
            Sb += len(idx) * (diff @ diff.T)

        A = np.linalg.pinv(Sw + 1e-6 * np.eye(n)) @ Sb
        eigvals, eigvecs = np.linalg.eig(A)
        idx = np.argsort(np.real(eigvals))[::-1]

        max_comp = min(self.n_components, len(classes) - 1)
        self.alphas = np.real(eigvecs[:, idx[:max_comp]])

        self.K_train = K
        self.K_row_mean = np.mean(self._kernel(X, X), axis=0)
        self.K_all_mean = np.mean(self._kernel(X, X))

    def transform(self, X):
        K = self._kernel(X, self.X)
        K_mean_rows = np.mean(K, axis=1, keepdims=True)
        Kc = K - self.K_row_mean.reshape(1, -1) - K_mean_rows + self.K_all_mean
        return Kc @ self.alphas


# Reconstruction helper for Fisherfaces
def fisher_reconstruct(x, pca, lda):
    x_pca = pca.transform(np.atleast_2d(x))
    x_lda = lda.transform(x_pca)
    x_pca_rec = x_lda @ lda.components.T + lda.mean
    x_img_rec = x_pca_rec @ pca.components.T + pca.mean
    return x_img_rec[0]


# Visuals and images
def show_faces(vecs, title):
    plt.figure(figsize=(10, 5))
    n = min(25, vecs.shape[1])
    for i in range(n):
        plt.subplot(5, 5, i + 1)
        img = np.real(vecs[:, i]).reshape(50, 50)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.suptitle(title)
    plt.show()


def show_face_grid(vecs, title, save_path):
    plt.figure(figsize=(10, 10))
    n = min(25, vecs.shape[1])
    for i in range(n):
        plt.subplot(5, 5, i + 1)
        face = np.real(vecs[:, i]).reshape(50, 50)
        face = face - np.mean(face)
        face = face / (np.std(face) + 1e-8)
        face = (face - np.min(face)) / (np.max(face) - np.min(face) + 1e-8)
        plt.imshow(face, cmap='gray')
        plt.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def show_reconstructions(pca, lda, X, save_path="reconstruction_faces.png"):
    idx = np.random.choice(len(X), 10, replace=False)
    plt.figure(figsize=(15, 5))

    for i in range(10):
        original = X[idx[i]].reshape(50, 50)
        eigen_rec = pca.reconstruct(X[idx[i]]).reshape(50, 50)
        fisher_rec = fisher_reconstruct(X[idx[i]], pca, lda).reshape(50, 50)

        plt.subplot(3, 10, i + 1)
        plt.imshow(original, cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title("Original")

        plt.subplot(3, 10, i + 11)
        plt.imshow(eigen_rec, cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title("Eigen")

        plt.subplot(3, 10, i + 21)
        plt.imshow(fisher_rec, cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title("Fisher")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def show_reference_style(basis_vecs, pca, lda, X, title, save_path, mode="eigen"):
    idx = np.random.choice(len(X), 10, replace=False)

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(title, fontsize=16)

    n = min(25, basis_vecs.shape[1])

    # left block: basis faces
    for i in range(n):
        ax = plt.subplot2grid((5, 10), (i // 5, i % 5))
        face = np.real(basis_vecs[:, i]).reshape(50, 50)
        face = face - np.mean(face)
        face = face / (np.std(face) + 1e-8)
        face = (face - np.min(face)) / (np.max(face) - np.min(face) + 1e-8)
        ax.imshow(face, cmap='gray')
        ax.axis('off')

    # right block: original faces
    for i in range(10):
        ax = plt.subplot2grid((5, 10), (2, i))
        ax.imshow(X[idx[i]].reshape(50, 50), cmap='gray')
        ax.axis('off')

    # right block: reconstructed faces
    for i in range(10):
        ax = plt.subplot2grid((5, 10), (3, i))
        if mode == "eigen":
            recon = pca.reconstruct(X[idx[i]]).reshape(50, 50)
        else:
            recon = fisher_reconstruct(X[idx[i]], pca, lda).reshape(50, 50)
        ax.imshow(recon, cmap='gray')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


# Main method
def main():
    X_train, y_train = CreateMatrix("./Yale_Face_Database/Training/")
    X_test, y_test = CreateMatrix("./Yale_Face_Database/Testing/")

    n_classes = len(np.unique(y_train))

    # PCA
    pca = PCA(n_components=100)
    pca.fit(X_train)

    # LDA trained on PCA features
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    lda = LDA(n_components=min(14, n_classes - 1))
    lda.fit(X_train_pca, y_train)

    # Fisherfaces back to image space
    fisherfaces = pca.components @ lda.components

    show_face_grid(pca.components, "Eigenfaces (PCA)", "eigenfaces_grid.png")
    show_face_grid(fisherfaces, "Fisherfaces (LDA)", "fisherfaces_grid.png")
    show_reconstructions(pca, lda, X_train)

    show_reference_style(
        pca.components,
        pca,
        lda,
        X_train,
        "Eigenfaces and its reconstruct faces",
        "eigenfaces_reference_style.png",
        mode="eigen"
    )

    show_reference_style(
        fisherfaces,
        pca,
        lda,
        X_train,
        "Fisherfaces and its reconstruct faces",
        "fisherfaces_reference_style.png",
        mode="fisher"
    )

    # classification
    knn = KNN(k=5)

    knn.fit(X_train_pca, y_train)
    pca_acc = np.mean(knn.predict(X_test_pca) == y_test)

    X_train_lda = lda.transform(X_train_pca)
    X_test_lda = lda.transform(X_test_pca)
    knn.fit(X_train_lda, y_train)
    lda_acc = np.mean(knn.predict(X_test_lda) == y_test)

    kpca_acc = []
    klda_acc = []

    # kernel methods
    for k in ["rbf", "linear"]:
        kpca = KernelPCA(kernel=k, n_components=50)
        kpca.fit(X_train)
        X_train_kpca = kpca.transform(X_train)
        X_test_kpca = kpca.transform(X_test)
        knn.fit(X_train_kpca, y_train)
        kpca_acc.append(np.mean(knn.predict(X_test_kpca) == y_test))

        klda = KernelLDA(kernel=k, n_components=min(10, n_classes - 1))
        klda.fit(X_train, y_train)
        X_train_klda = klda.transform(X_train)
        X_test_klda = klda.transform(X_test)
        knn.fit(X_train_klda, y_train)
        klda_acc.append(np.mean(knn.predict(X_test_klda) == y_test))

    print("\n===== FINAL RESULTS =====")
    print(f"PCA Accuracy: {pca_acc:.3f}")
    print(f"LDA Accuracy: {lda_acc:.3f}")
    print(f"Kernel PCA (RBF): {kpca_acc[0]:.3f}")
    print(f"Kernel PCA (Linear): {kpca_acc[1]:.3f}")
    print(f"Kernel LDA (RBF): {klda_acc[0]:.3f}")
    print(f"Kernel LDA (Linear): {klda_acc[1]:.3f}")
    print(f"Number of classes: {n_classes}")
    print(f"Maximum Fisherfaces possible: {n_classes - 1}")


if __name__ == "__main__":
    main()