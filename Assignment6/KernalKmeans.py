from PIL import Image
import numpy as np
from scipy.spatial.distance import cdist
import os


IMAGE_PATH = ["image1.png","image2.png"]
GAMMA_S = 0.001
GAMMA_C = 0.001
NUM_CLUSTERS = 4
MAX_ITER = 100
OUTPUT_DIR = "kernel_kmeans_output"
np.random.seed(42)
os.makedirs(OUTPUT_DIR, exist_ok=True)



def build_kernel(coords, colors, gamma_s, gamma_c):
    spatial_dist2 = cdist(coords,coords,metric="sqeuclidean")
    color_dist2 = cdist(colors,colors,metric="sqeuclidean")

    kernel = (np.exp(-gamma_s * spatial_dist2)*np.exp(-gamma_c * color_dist2))
    return kernel.astype(np.float32)

# K-MEANS++ INITIALIZATION - K++ means


# def initialize_clusters(coords, colors, n_clusters):
#     coords_norm = coords / 99.0
#     colors_norm = colors / 255.0
#     features = np.hstack([coords_norm,colors_norm])
#     n = features.shape[0]
#     centers = []
#     first_center = np.random.randint(n)
#     centers.append(first_center)
#     for _ in range(n_clusters - 1):
#         min_dist = np.full(n, np.inf)
#         for c in centers:
#             dist = np.sum((features - features[c]) ** 2,axis=1)
#             min_dist = np.minimum(min_dist,dist)
#         total = np.sum(min_dist)
#         if total == 0:
#             next_center = np.random.randint(n)
#         else:
#             prob = min_dist / total
#             next_center = np.random.choice(n,p=prob)
#         centers.append(next_center)
#     centers = np.array(centers)
#     distances = []
#     for c in centers:
#         d = np.sum((features - features[c]) ** 2,axis=1)
#         distances.append(d)
#     distances = np.array(distances)
#     labels = np.argmin(distances,axis=0)
#     return labels

# Random Initialisation

def initialize_clusters(coords, colors, n_clusters):
    coords_norm = coords / 99.0
    colors_norm = colors / 255.0
    features = np.hstack([coords_norm, colors_norm])
    n = features.shape[0]
    centers = np.random.choice(n,size=n_clusters,replace=False)
    distances = []
    for c in centers:
        d = np.sum((features - features[c]) ** 2,axis=1)
        distances.append(d)
    distances = np.array(distances)
    labels = np.argmin(distances, axis=0 )
    return labels

# Kernal Distance

def compute_distance(kernel,kernel_diag,labels,cluster_id):
    cluster = np.where(labels == cluster_id)[0]
    nk = len(cluster)
    if nk == 0:
        return np.full(kernel.shape[0],np.inf)
    # K(x_i, x_i)
    term1 = kernel_diag
    # (2 / |Ck|) * sum K(x_i, x_j)
    term2 = (2.0 / nk) * np.sum(kernel[:, cluster],axis=1)
    # (1 / |Ck|²) * sum K(x_p, x_q)
    term3 = (1.0 / (nk * nk)) * np.sum(kernel[np.ix_(cluster,cluster)])
    return term1 - term2 + term3


def save_segmentation(labels,image_shape,filename):
    cluster_colors = np.array([[255, 0, 0],[0, 255, 0],[0, 0, 255],[255, 255, 0],[255, 0, 255],[0, 255, 255]])
    rgb = cluster_colors[labels % len(cluster_colors)]
    rgb = rgb.reshape(image_shape[0],image_shape[1],3)
    img = Image.fromarray(rgb.astype(np.uint8))
    img.save(filename)

def kernel_kmeans(kernel,labels,n_clusters,image_shape,max_iter=100):
    kernel_diag = np.diag(kernel)
    frame_files = []
    first_frame = os.path.join(OUTPUT_DIR,"frame_000.png")
    save_segmentation(labels,image_shape,first_frame)
    frame_files.append(first_frame)
    for iteration in range(max_iter):
        distances = []
        for k in range(n_clusters):
            d = compute_distance(kernel,kernel_diag,labels,k)
            distances.append(d)
        distances = np.array(distances)
        new_labels = np.argmin( distances,axis=0)
        filename = os.path.join(OUTPUT_DIR,f"frame_{iteration+1:03d}.png")
        save_segmentation(new_labels,image_shape,filename)
        frame_files.append(filename)
        changed = np.sum(labels != new_labels)
        print(f"Iteration {iteration+1}: "f"{changed} pixels changed")
        if np.array_equal(labels,new_labels):
            break
        labels = new_labels
    return labels, frame_files


def create_gif(frame_files, output_name):
    gif_dir = os.path.join(os.path.dirname(output_name), "gif")
    os.makedirs(gif_dir, exist_ok=True)
    output_path = os.path.join(gif_dir,os.path.basename(output_name))
    frames = [Image.open(file) for file in frame_files]
    frames[0].save(output_path,format="GIF",append_images=frames[1:],save_all=True,duration=500,loop=2)

# Main function
for i in IMAGE_PATH:
    img = np.array(Image.open(i))
    h, w, _ = img.shape
    coords = np.array([[i, j]for i in range(h)for j in range(w)])
    colors = img.reshape(-1, 3)
    print("Building kernel matrix")
    kernel = build_kernel(coords,colors,GAMMA_S,GAMMA_C)
    print("Initializing clusters")
    labels = initialize_clusters(coords,colors,NUM_CLUSTERS)

    print("Running Kernel K-Means")
    final_labels, frame_files = kernel_kmeans(kernel,labels,NUM_CLUSTERS,(h, w),MAX_ITER)
    print("Creating GIF")
    create_gif(frame_files, os.path.join(OUTPUT_DIR,f"kernel_kmeans{i}.gif"))
    print("Finished")
