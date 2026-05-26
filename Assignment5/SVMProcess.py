import numpy as np
import pandas as pandas
from libsvm.svmutil import *
import sys

# 1. data loader for csv
def load_data(x_path, y_path):
    X = pandas.read_csv(x_path, header=None).values
    Y = pandas.read_csv(y_path, header=None).values.ravel()
    return Y.tolist(), X.tolist()

# 2. normalization
def normalize(X):
    X = np.array(X, dtype=float)
    mn = X.min(axis=0)
    mx = X.max(axis=0)
    X = (X - mn) / (mx - mn + 1e-8)
    return X.tolist(), mn, mx

def normalize_test(X, mn, mx):
    X = np.array(X, dtype=float)
    X = (X - mn) / (mx - mn + 1e-8)
    return X.tolist()

# 3. k fold split
def kfold_split(X, Y, k=3):
    X = np.array(X)
    Y = np.array(Y)
    index = np.arange(len(X))
    np.random.shuffle(index)
    fold_size = len(X) // k
    folds = []
    for i in range(k):
        if i == k - 1:
            val_index = index[i*fold_size:]
        else:
            val_index = index[i*fold_size:(i+1)*fold_size]
        train_index = np.setdiff1d(index, val_index)
        folds.append((train_index, val_index))

    return folds

# 4. grid search for various values of C and gamma
def grid_search(X, Y, kernel_type):
    C_list = [0.1, 1, 10, 100]
    gamma_list = [1e-3, 1e-4, 1e-5]
    best_acc = 0
    best_params = None
    folds = kfold_split(X, Y, k=3)
    for C in C_list:
        # linear
        if kernel_type == 0:
            accs = []
            for tr_idx, val_idx in folds:
                X_tr = [X[i] for i in tr_idx]
                Y_tr = [Y[i] for i in tr_idx]
                X_val = [X[i] for i in val_idx]
                Y_val = [Y[i] for i in val_idx]
                model = svm_train(Y_tr,X_tr,f'-t 0 -c {C} -q')
                _, acc, _ = svm_predict(Y_val,X_val,model,options='-q')
                accs.append(acc[0])
            mean_acc = np.mean(accs)
            print(f"Linear -> C={C}, acc={mean_acc}")
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_params = C
        # polynomial/RBF
        else:
            for g in gamma_list:
                accs = []
                for tr_idx, val_idx in folds:
                    X_tr = [X[i] for i in tr_idx]
                    Y_tr = [Y[i] for i in tr_idx]
                    X_val = [X[i] for i in val_idx]
                    Y_val = [Y[i] for i in val_idx]
                    # polynomial
                    if kernel_type == 1:
                        model = svm_train(Y_tr,X_tr,f'-t 1 -c {C} -g {g} -d 3 -r 1 -q')
                    # RBF
                    elif kernel_type == 2:
                        model = svm_train(Y_tr,X_tr,f'-t 2 -c {C} -g {g} -q')
                    _, acc, _ = svm_predict(Y_val,X_val,model,options='-q')
                    accs.append(acc[0])
                mean_acc = np.mean(accs)

                if kernel_type == 1:
                    print(f"Poly -> C={C}, gamma={g}, acc={mean_acc}")
                elif kernel_type == 2:
                    print(f"RBF -> C={C}, gamma={g}, acc={mean_acc}")
                if mean_acc > best_acc:
                    best_acc = mean_acc
                    best_params = (C, g)

    return best_params, best_acc


# 5. all kernals of SVM
def linear_kernel(X1, X2):
    return np.dot(np.array(X1), np.array(X2).T)

def rbf_kernel(X1, X2, gamma=0.001):
    X1 = np.array(X1)
    X2 = np.array(X2)
    X1_sq = np.sum(X1**2, axis=1).reshape(-1, 1)
    X2_sq = np.sum(X2**2, axis=1)
    dist = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)
    return np.exp(-gamma * dist)

def combined_kernel(X1, X2, alpha=0.5, gamma=0.001):
    return alpha * linear_kernel(X1, X2) + (1 - alpha) * rbf_kernel(X1, X2, gamma)

# 6. precomputed kernal format
def to_precomputed(K):
    n = K.shape[0]
    newK = []
    for i in range(n):
        newK.append([i + 1] + K[i].tolist())
    return newK
# 7. pipeline for all parts
if __name__ == "__main__":
    # data loader
    Y_train, X_train = load_data("./data/X_train.csv", "./data/Y_train.csv")
    Y_test, X_test = load_data("./data/X_test.csv", "./data/Y_test.csv")

    # normalising the data
    X_train, mn, mx = normalize(X_train)
    X_test = normalize_test(X_test, mn, mx)

    print("Data loaded and normalized")

    # PART 1: all 3 models default params
    print("\nLinear SVM")
    model_linear = svm_train(Y_train, X_train, '-t 0 -c 1 -q')
    _,acc_linear,_= svm_predict(Y_test, X_test, model_linear)

    print("\nPolynomial SVM")
    model_poly = svm_train(Y_train, X_train, '-t 1 -c 1 -d 3 -g 1 -r 1 -q')
    _,acc_polynomial,_=svm_predict(Y_test, X_test, model_poly)

    print("\nRBF SVM model")
    model_rbf = svm_train(Y_train, X_train, '-t 2 -c 1 -g 0.001 -q')
    _,acc_rbf,_=svm_predict(Y_test, X_test, model_rbf)
    #Compariosn of above 3
    print("Accuracy comparison of Linear, Polynomial and RBF models")
    print(f'Linear SVM : {acc_linear[0]}\nPolynomial SVM: {acc_polynomial[0]}\nRadial basis SVM: {acc_rbf[0]}\n')


    # PART 2: grid search for (C,gamma)
    results=[]
    print("\n--- TUNING LINEAR SVM ---")
    best_C_linear, _= grid_search(X_train, Y_train, 0)
    model_linear = svm_train(Y_train,X_train,f'-t 0 -c {best_C_linear} -q')
    _, acc_linear, _ = svm_predict(Y_test,X_test,model_linear)
    results.append(["Linear", acc_linear[0]])

    print("\n--- TUNING POLY SVM ---")
    (best_C_poly, best_g_poly), _ = grid_search(X_train,Y_train,1)
    model_poly = svm_train(Y_train,X_train,f'-t 1 -c {best_C_poly} -g {best_g_poly} -d 3 -r 1 -q')
    _, acc_poly, _ = svm_predict(Y_test,X_test,model_poly)
    results.append(["Polynomial", acc_poly[0]])

    print("\n--- TUNING RBF SVM ---")
    (best_C_rbf, best_g_rbf), _ = grid_search(X_train,Y_train,2)
    model_rbf = svm_train(Y_train,X_train,f'-t 2 -c {best_C_rbf} -g {best_g_rbf} -q')
    _, acc_rbf, _ = svm_predict(Y_test,X_test,model_rbf)
    results.append(["RBF", acc_rbf[0]])


    # PART 3: custom kernal using linear and rbf
    # =========================
    print("\ncustom kernal using linear and rbf")
    K_train = combined_kernel(X_train, X_train)
    K_test = combined_kernel(X_test, X_train)
    K_train = to_precomputed(np.array(K_train))
    K_test = to_precomputed(np.array(K_test))
    model_custom = svm_train(Y_train, K_train, '-t 4 -c 1')
    svm_predict(Y_test, K_test, model_custom)