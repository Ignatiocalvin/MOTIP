import os
import cv2
import argparse
from tqdm import tqdm

# --- Define File Types ---
# Note: The original script had these swapped. I have corrected them.
ANN_TYPE = "txt"
VIDEO_TYPE = "MP4"


def main(args):
    """
    Main function to run the extraction.
    """
    # 1. Find all matching pairs
    output_seq_dir = os.path.join('images', '10-07-2019-1-5', "img1")

    vidcap = cv2.VideoCapture(args.video_path)

    if not vidcap.isOpened():
        print(f"  Error: Could not open video file: {args.video_path}, skipping.")
        return
    
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
        img_path = os.path.join(output_seq_dir, img_name)
        
        cv2.imwrite(img_path, image)

    vidcap.release()
        
    print("Frame extraction complete.")
    print(f"All frames have been saved to: {args.converted_img_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P-DESTRE Video to Frame Extractor for MOTIP")
    
    # --- TODO: SET YOUR PATHS HERE or via command line ---
    
    # The root directory of the P-DESTRE dataset
    DEFAULT_DATA_ROOT = "./P-DESTRE"

    parser.add_argument('--video_path', type=str,
                        default=os.path.join(DEFAULT_DATA_ROOT, 'videos'),
                        help="Path to the folder with P-DESTRE.MP4 videos.")
    
    args = parser.parse_args()
    
    main(args)