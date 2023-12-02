import os
import cv2
import numpy as np

# Initialize SIFT detector
sift = cv2.SIFT_create()

def get_video_descriptors(video_path):
    """
    Extract SIFT descriptors from a video.
    """
    descriptors_list = []

    # Read the video
    cap = cv2.VideoCapture(video_path)
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break

        # Convert to grayscale for SIFT
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Extract keypoints and descriptors
        _, descriptors = sift.detectAndCompute(gray, None)
        
        if descriptors is not None:
            descriptors_list.extend(descriptors)

    cap.release()
    return descriptors_list

def save_descriptors_npy(descriptors, save_path):
    """
    Save descriptors as a .npy file
    """
    np.save(save_path, descriptors)

def main():
    root_folder = "merged"
    output_root_folder = "merged_descriptors"
    
    # Ensure the output directory exists
    if not os.path.exists(output_root_folder):
        os.makedirs(output_root_folder)

    # Traverse through all the folders inside 'merged'
    for action_class in os.listdir(root_folder):
        class_folder = os.path.join(root_folder, action_class)
        output_class_folder = os.path.join(output_root_folder, action_class)

        # Ensure the output class directory exists
        if not os.path.exists(output_class_folder):
            os.makedirs(output_class_folder)

        if os.path.isdir(class_folder):
            for video_file in os.listdir(class_folder):
                video_path = os.path.join(class_folder, video_file)
                
                # Ensure it's a video file by checking the extension
                if video_path.endswith(('.mp4', '.avi')):
                    descriptors = get_video_descriptors(video_path)
                    
                    # Save descriptors to .npy format
                    video_name_without_ext = os.path.splitext(video_file)[0]
                    save_path = os.path.join(output_class_folder, f"{video_name_without_ext}.npy")
                    save_descriptors_npy(descriptors, save_path)

                    print(f"Saved descriptors of {video_file} in class {action_class} to {save_path}")

if __name__ == "__main__":
    main()
