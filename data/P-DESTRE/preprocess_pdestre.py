import os
import cv2
import argparse
from tqdm import tqdm

# --- Define File Types ---
# Note: The original script had these swapped. I have corrected them.
ANN_TYPE = "txt"
VIDEO_TYPE = "MP4"

def check_pairs_in_dataset(ann_dir, video_dir):
    """
    Checks for paired annotation and video names.

    Args:
        ann_dir (str): A path to the folder with annotations.
        video_dir (str): A path to folder with videos.

    Returns:
         name_pairs: A list of strings (names of files that are paired).
    """

    # --- This is how you get the list of annotation files ---
    try:
        # List all files in the annotation directory
        annotation_names = os.listdir(ann_dir)
    except FileNotFoundError:
        print(f"Error: Annotation directory not found at {ann_dir}")
        return
    # --- End of code ---

    # --- This is how you get the list of video files ---
    try:
        # List all files in the video directory
        video_names = os.listdir(video_dir)
    except FileNotFoundError:
        print(f"Error: Video directory not found at {video_dir}")
        return
    # --- End of code ---

    # remove.type suffix
    annotation_type = f".{ANN_TYPE}"
    video_type = f".{VIDEO_TYPE}"
    
    # This loop correctly removes the suffix (e.g., ".txt")
    for i, ann in enumerate(annotation_names):
        annotation_names[i] = ann.removesuffix(annotation_type)
    
    # This loop correctly removes the suffix (e.g., ".MP4")
    for i, vid in enumerate(video_names):
        video_names[i] = vid.removesuffix(video_type)

    name_pairs = []
    for annotation_name in annotation_names:
        if annotation_name in video_names:
            # remove wrong data pairs:
            if not annotation_name.startswith("._"):
                name_pairs.append(annotation_name)

    return name_pairs

def extract_frames_for_sequence(video_path, output_image_dir):
    """
    Extracts all frames from a single video and saves them to the
    target directory in the format 'motip' expects (img1/000001.jpg,...).

    Args:
        video_path (str): Path to the source.MP4 video file.
        output_image_dir (str): Path to the destination folder, 
                                e.g., ".../images/08-11-2019-1-1/img1"
    """
    
    if not os.path.exists(video_path):
        print(f"  Warning: Video file not found: {video_path}, skipping.")
        return

    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        print(f"  Error: Could not open video file: {video_path}, skipping.")
        return

    # Create the 'img1' directory
    os.makedirs(output_image_dir, exist_ok=True)
    
    frame_idx = 0
    while True:
        success, image = vidcap.read()
        if not success:
            break  # End of video
        
        # P-DESTRE annotations are 1-indexed. Our loop is 0-indexed.
        # So, the first frame read (idx 0) is frame 1.
        frame_idx += 1
        
        # Format the image name as 000001.jpg, 000002.jpg,...
        # This 6-digit padding matches the 'DanceTrack' format.
        img_name = f"{frame_idx:06d}.jpg"
        img_path = os.path.join(output_image_dir, img_name)
        
        cv2.imwrite(img_path, image)

    vidcap.release()
    # print(f"  Successfully extracted {frame_idx} frames.")

def main(args):
    """
    Main function to run the extraction.
    """
    # 1. Find all matching pairs
    names_paired = check_pairs_in_dataset(args.ann_dir, args.video_dir)
    
    if not names_paired:
        print("No matching video/annotation pairs found. Exiting.")
        return

    print(f"Starting frame extraction for {len(names_paired)} sequences...")
    
    # 2. Loop through each pair and extract frames
    for name in tqdm(names_paired):
        print(f"Processing sequence: {name}")
        # Define the source video path
        video_path = os.path.join(args.video_dir, f"{name}.{VIDEO_TYPE}")
        
        # Define the target directory in the 'motip'/'DanceTrack' format
        # e.g., P-DESTRE/images/08-11-2019-1-1/img1/
        output_seq_dir = os.path.join(args.converted_img_dir, name, "img1")
        
        # Extract all frames
        extract_frames_for_sequence(video_path, output_seq_dir)
        
    print("Frame extraction complete.")
    print(f"All frames have been saved to: {args.converted_img_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P-DESTRE Video to Frame Extractor for MOTIP")
    
    # --- TODO: SET YOUR PATHS HERE or via command line ---
    
    # The root directory of the P-DESTRE dataset
    DEFAULT_DATA_ROOT = "./"
    
    parser.add_argument('--ann_dir', type=str, 
                        default=os.path.join(DEFAULT_DATA_ROOT, 'annotations'),
                        help="Path to the folder with P-DESTRE.txt annotations.")
                        
    parser.add_argument('--video_dir', type=str, 
                        default=os.path.join(DEFAULT_DATA_ROOT, 'videos'),
                        help="Path to the folder with P-DESTRE.MP4 videos.")
                        
    parser.add_argument('--converted_img_dir', type=str, 
                        default=os.path.join(DEFAULT_DATA_ROOT, 'images'),
                        help="Path to the *output* folder where extracted frames will be saved.")
    
    args = parser.parse_args()
    
    main(args)