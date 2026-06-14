import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


# ============================================================
# LOAD DATA
# ============================================================
labels = np.loadtxt("./MNSIT database/mnist2500_labels.txt")


def make_gif(history, labels, filename, folder):
    frames = []
    os.makedirs(folder, exist_ok=True)
    for i, Y in enumerate(history):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(Y[:, 0],Y[:, 1],c=labels,cmap="tab10",s=5)
        ax.set_title(f"Iteration {i * 10}")
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        frame_path = os.path.join(folder, f"frame_{i}.png")
        plt.savefig(frame_path, dpi=150)
        plt.close(fig)
        frames.append(Image.open(frame_path))
    frames[0].save(filename,save_all=True,append_images=frames[1:],duration=300,loop=0)
    print("Saved:", filename)
# ============================================================
# PART 3: P vs Q DISTRIBUTION
# ============================================================
def plot_similarity_histograms(perplexity):
    P_tsne = np.load(f"./npy_files/P_tsne{perplexity}.npy")
    Q_tsne = np.load(f"./npy_files/Q_tsne{perplexity}.npy")
    P_sne = np.load(f"./npy_files/P_sne{perplexity}.npy")
    Q_sne = np.load(f"./npy_files/Q_sne{perplexity}.npy")

    # remove diagonal zeros if matrices are NxN
    P_tsne = P_tsne[P_tsne > 0]
    Q_tsne = Q_tsne[Q_tsne > 0]
    P_sne = P_sne[P_sne > 0]
    Q_sne = Q_sne[Q_sne > 0]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # --------------------
    # High-D (P)
    # --------------------
    axes[0, 0].hist(P_tsne.flatten(), bins=50)
    axes[0, 0].set_title("t-SNE")
    axes[0, 0].set_ylabel("High-D")
    axes[0, 0].set_yscale("log")

    axes[0, 1].hist(P_sne.flatten(), bins=50)
    axes[0, 1].set_title("Symmetric SNE")
    axes[0, 1].set_yscale("log")

    # --------------------
    # Low-D (Q)
    # --------------------
    axes[1, 0].hist(Q_tsne.flatten(), bins=50)
    axes[1, 0].set_ylabel("Low-D")
    axes[1, 0].set_yscale("log")

    axes[1, 1].hist(Q_sne.flatten(), bins=50)
    axes[1, 1].set_yscale("log")

    plt.suptitle(f"Pairwise Similarity Distribution (Perplexity = {perplexity})")

    plt.tight_layout()
    plt.savefig(f"./visual_tsne/similarity_histogram_{perplexity}.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    perplexity = [15,20,50,65]
    for prep in perplexity:
        history_tsne = np.load(f"./npy_files/history_tsne{prep}.npy", allow_pickle=True)
        history_sne = np.load(f"./npy_files/history_sne{prep}.npy", allow_pickle=True)
        make_gif(history_tsne, labels,f"./visual_tsne/tsne_{prep}.gif",f"./Frames/tsneFrames_{prep}")
        make_gif(history_sne, labels,f"./visual_tsne/sne_{prep}.gif",f"./Frames/sneFrames_{prep}")
    for p in perplexity:
        plot_similarity_histograms(p)