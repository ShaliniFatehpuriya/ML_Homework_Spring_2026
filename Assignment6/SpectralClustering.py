from PIL import Image
import numpy as np
from scipy.spatial.distance import cdist
import os

IMAGE_PATH = ["image1.png","image2.png"]
GAMMA_S = 0.0001
GAMMA_C = 0.001
NUM_CLUSTERS = 2
MAX_ITER = 100
MODE = "normalized"      # "ratio" or "normalized"
OUTPUT_DIR = f"spectral_{MODE}_output"
np.random.seed(42)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_kernel(coords, colors, gamma_s, gamma_c):
    spatial_dist2 = cdist(coords,coords,metric="sqeuclidean").astype(np.float32)
    color_dist2 = cdist(colors,colors,metric="sqeuclidean").astype(np.float32)
    kernel = np.exp(-gamma_s * spatial_dist2,dtype=np.float32)
    del spatial_dist2
    kernel *= np.exp(-gamma_c * color_dist2,dtype=np.float32)
    del color_dist2
    return kernel

def ratio_cut_laplacian(W):
    degree = np.sum(W, axis=1)
    L = -W.copy()
    np.fill_diagonal(L, degree)
    return L

def normalized_cut_laplacian(W):
    degree = np.sum(W, axis=1)
    D_inv_sqrt = 1.0 / np.sqrt(degree + 1e-12)
    L = -W.copy()
    np.fill_diagonal(L, degree)
    L *= D_inv_sqrt[:, None]
    L *= D_inv_sqrt[None, :]
    return L

def compute_eigenvectors(L, n_clusters):
    print("Computing eigenvectors")
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    U = eigenvectors[:,1:n_clusters+1].copy()
    del eigenvectors
    del eigenvalues
    return U.astype(np.float32)


def normalize_rows(U):
    norms = np.linalg.norm(U,axis=1,keepdims=True)
    return U / (norms + 1e-12)

# K++means initialisation
# def initialize_clusters(features,n_clusters):
#     n = features.shape[0]
#     centers = []
#     first_center = np.random.randint(n)
#     centers.append(first_center)
#     for _ in range(n_clusters - 1):
#         min_dist = np.full( n,np.inf)
#         for c in centers:
#             dist = np.sum((features - features[c])**2,axis=1)
#             min_dist = np.minimum(min_dist,dist)
#         total = np.sum(min_dist)
#         if total == 0:
#             next_center = np.random.randint(n)
#         else:
#             prob = min_dist / total
#             next_center = np.random.choice(n,p=prob)
#         centers.append(next_center)
#     centers = np.array(centers,dtype=np.int32)
#     distances = []
#     for c in centers:
#         d = np.sum((features - features[c])**2,axis=1)
#         distances.append(d)
#     distances = np.array(distances)
#     labels = np.argmin(distances,axis=0)
#     return labels


# Random initialisation
def initialize_clusters(features, n_clusters):
    n = features.shape[0]
    # Randomly choose k points as initial centers
    centers = np.random.choice(n, size=n_clusters, replace=False)
    # Compute distance from every point to every center
    distances = []
    for c in centers:
        d = np.sum((features - features[c]) ** 2, axis=1)
        distances.append(d)
    distances = np.array(distances)
    # Assign each point to nearest center
    labels = np.argmin(distances, axis=0)
    return labels

def kmeans(features,labels,n_clusters,image_shape,max_iter=100):
    frame_files = []
    first_frame = os.path.join(OUTPUT_DIR,"frame_000.png")
    save_segmentation(labels,image_shape,first_frame)
    frame_files.append(first_frame)
    for iteration in range(max_iter):
        centers = []
        for k in range(n_clusters):
            cluster = features[labels == k]
            if len(cluster) == 0:
                centers.append(np.zeros(features.shape[1],dtype=np.float32))
            else:
                centers.append(np.mean(cluster,axis=0))
        centers = np.array( centers,dtype=np.float32)
        distances = cdist(features,centers,metric="sqeuclidean")
        new_labels = np.argmin(distances,axis=1)
        filename = os.path.join(OUTPUT_DIR,f"frame_{iteration+1:03d}.png")
        save_segmentation(new_labels,image_shape,filename)
        frame_files.append(filename)
        changed = np.sum(labels != new_labels)
        print(f"Iteration {iteration+1}: "f"{changed} pixels changed")
        if np.array_equal(labels,new_labels):
            break
        labels = new_labels
    return labels, frame_files


def save_segmentation(labels,image_shape,filename):
    cluster_colors = np.array([[255, 0, 0],[0, 255, 0],[0, 0, 255],[255, 255, 0],[255, 0, 255],[0, 255, 255]])
    rgb = cluster_colors[labels % len(cluster_colors)]
    rgb = rgb.reshape(image_shape[0],image_shape[1],3)
    img = Image.fromarray(rgb.astype(np.uint8))
    img.save(filename)

def create_gif(frame_files, output_name):
    gif_dir = os.path.join(os.path.dirname(output_name), "gif")
    os.makedirs(gif_dir, exist_ok=True)
    output_path = os.path.join(gif_dir,os.path.basename(output_name))
    frames = []
    for file in frame_files:
        img = Image.open(file)
        frames.append(img.copy())
        img.close()
    frames[0].save(output_path,format="GIF",append_images=frames[1:],save_all=True,duration=500,loop=2)



import matplotlib.pyplot as plt
# For K=2
def plot_eigenspace(U, labels, filename):
    plt.figure(figsize=(6,6))
    for k in range(NUM_CLUSTERS):
        cluster = U[labels == k]
        plt.scatter(
            cluster[:,0],
            cluster[:,1],
            s=2,
            label=f"Cluster {k}"
        )

    plt.xlabel("Eigenvector 1")
    plt.ylabel("Eigenvector 2")
    plt.legend()
    plt.title("Spectral Embedding")
    plt.savefig(filename)
    plt.close()
# For K=3
from mpl_toolkits.mplot3d import Axes3D

def plot_eigenspace_3d(U, labels, filename):
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')
    for k in range(NUM_CLUSTERS):
        cluster = U[labels == k]
        ax.scatter(cluster[:,0],cluster[:,1],cluster[:,2],s=2,label=f"Cluster {k}")
    ax.set_xlabel("Eigenvector 1")
    ax.set_ylabel("Eigenvector 2")
    ax.set_zlabel("Eigenvector 3")
    plt.legend()
    plt.savefig(filename)
    plt.close()


#-----------------Main--------------------#
for IMAGE_PATH in IMAGE_PATH:
    img = np.array(Image.open(IMAGE_PATH))
    h, w, _ = img.shape
    coords = np.array([[i, j]for i in range(h)for j in range(w)],dtype=np.float32)
    colors = img.reshape(-1,3).astype(np.float32)
    print("Building kernel...")
    W = build_kernel(coords,colors,GAMMA_S,GAMMA_C)
    print("Building Laplacian...")
    if MODE == "ratio":
        L = ratio_cut_laplacian(W)
    elif MODE == "normalized":
        L = normalized_cut_laplacian(W)
    else:
        raise ValueError("mode should be ratio or normalized")
    del W
    U = compute_eigenvectors(L,NUM_CLUSTERS)
    del L
    if MODE == "normalized":
        U = normalize_rows(U)
    print("Initializing clusters")
    labels = initialize_clusters(U,NUM_CLUSTERS)
    print("Cluster sizes:",np.bincount(labels))
    print("Running spectral clustering")
    final_labels, frame_files = kmeans(U,labels,NUM_CLUSTERS,(h, w),MAX_ITER)
    plot_eigenspace(U,final_labels,os.path.join(OUTPUT_DIR,f"eigenspace{IMAGE_PATH}.png"))
    print("Creating GIF")
    create_gif(frame_files,os.path.join(OUTPUT_DIR,f"spectral_{IMAGE_PATH}.gif"))
    print("Finished")