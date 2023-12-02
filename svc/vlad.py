import numpy as np
from sklearn.cluster import KMeans
import os
from tqdm import tqdm

def load_npy_descriptors(folder):
    """Load .npy descriptor files from a folder."""
    all_descriptors = []
    for file_name in os.listdir(folder):
        if file_name.endswith('.npy'):
            descriptors = np.load(os.path.join(folder, file_name))
            all_descriptors.extend(descriptors)
    return np.array(all_descriptors)

def vlad_encoding(descriptor_list, kmeans):
    """Encode descriptors using the VLAD method."""
    vlad_vector = np.zeros((kmeans.n_clusters, descriptor_list.shape[1]))
    predicted_labels = kmeans.predict(descriptor_list)
    for i, descriptor in enumerate(descriptor_list):
        cluster_idx = predicted_labels[i]
        vlad_vector[cluster_idx] += descriptor - kmeans.cluster_centers_[cluster_idx]
    vlad_vector = vlad_vector.flatten()
    # L2 normalization
    vlad_vector = vlad_vector / np.sqrt(np.dot(vlad_vector, vlad_vector))
    return vlad_vector

def main():
    root_folder = "merged_descriptors"
    output_root_folder = "vlad_representation"
    num_clusters = 64  # Typically smaller than BoVW

    # Ensure the output directory exists
    if not os.path.exists(output_root_folder):
        os.makedirs(output_root_folder)

    # 1. Load all descriptors with progress bar
    all_descriptors = []
    for action_class in tqdm(os.listdir(root_folder), desc="Loading Descriptors", position=0, leave=True):
        class_folder = os.path.join(root_folder, action_class)
        all_descriptors.extend(load_npy_descriptors(class_folder))
    
    # 2. Perform KMeans clustering for codebook generation with progress
    print("Performing KMeans clustering...")
    kmeans = KMeans(n_clusters=num_clusters).fit(np.vstack(all_descriptors).astype(np.float32))
    kmeans.cluster_centers_ = kmeans.cluster_centers_.astype(np.float32)

    # 3. VLAD encoding for each video
    for action_class in tqdm(os.listdir(root_folder), desc="VLAD Encoding", position=0, leave=True):
        class_folder = os.path.join(root_folder, action_class)
        output_class_folder = os.path.join(output_root_folder, action_class)
        if not os.path.exists(output_class_folder):
            os.makedirs(output_class_folder)

        for file_name in os.listdir(class_folder):
            if file_name.endswith('.npy'):
                descriptor_list = np.load(os.path.join(class_folder, file_name)).astype(np.float32)
                vlad_vector = vlad_encoding(descriptor_list, kmeans)

                # Save VLAD vector as .npy
                save_path = os.path.join(output_class_folder, file_name)
                np.save(save_path, vlad_vector)
                print(f"Saved VLAD encoding of {file_name} in class {action_class} to {save_path}")

if __name__ == "__main__":
    main()
