# Copyright (c) Ruopeng Gao. All Rights Reserved.

import os
import torch
from configparser import ConfigParser
from collections import defaultdict

from .dancetrack import DanceTrack
from .util import is_legal, append_annotation


class PDESTRE(DanceTrack):
    def __init__(
            self,
            data_root: str = "./data/",
            sub_dir: str = "P-DESTRE",  # Directory name for P-DESTRE
            split: str = "train",
            load_annotation: bool = True,
    ):
        # Path to the folder containing Train_0.txt, Test_0.txt, etc.
        self.split_files_dir = os.path.join(data_root, sub_dir, "splits")
        
        # Set these *before* parent __init__ so _get_sequence_infos can use them
        self.data_root = data_root
        self.sub_dir = sub_dir
        self.split = split

        # We must get sequence infos *before* calling the parent init,
        # as the parent init will call _get_sequence_names, which
        # now relies on sequence_infos.
        self.sequence_infos = self._get_sequence_infos()
        
        # Now call the parent __init__
        super(PDESTRE, self).__init__(
            data_root=data_root,
            sub_dir=sub_dir,
            split=split,
            load_annotation=load_annotation,
        )

    def _get_all_sequence_names_from_splits(self) -> list:
        """
        Gets all sequence names by reading the Train_*.txt, Test_*.txt, 
        or val_*.txt files from the train_test_splits directory.
        """
        all_sequence_names = []
        if not os.path.exists(self.split_files_dir):
            raise FileNotFoundError(
                f"P-DESTRE split file directory not found at: {self.split_files_dir}\n"
                f"Please ensure Train_*.txt, Test_*.txt, etc. are placed here."
            )

        split_files = []
        default_splits = ["train", "val", "test"]
        
        current_split_name = self.split.lower()
        
        if current_split_name in default_splits:
            # --- Default Logic: Find all files with the prefix ---
            split_prefix = current_split_name.capitalize()
            if current_split_name == "val":
                split_prefix = "val" # P-DESTRE uses "val_0.txt"
            
            split_files = [f for f in os.listdir(self.split_files_dir) 
                           if f.startswith(split_prefix) and f.endswith(".txt")]
        else:
            # --- Custom Logic: Find the exact file name ---
            custom_file = f"{self.split}.txt"
            if os.path.exists(os.path.join(self.split_files_dir, custom_file)):
                split_files = [custom_file]
            else:
                raise FileNotFoundError(
                    f"Custom split file '{custom_file}' not found in {self.split_files_dir}."
                    f"Make sure your DATASET_SPLITS in the.yaml config matches the file name."
                )

        if not split_files:
            raise FileNotFoundError(
                f"No split files found for split '{self.split}' in {self.split_files_dir}"
            )

        for file_name in split_files:
            with open(os.path.join(self.split_files_dir, file_name), "r") as f:
                for line in f:
                    seq_name = line.strip()
                    if not seq_name:
                        continue
                    
                    if seq_name.endswith(".txt"):
                        seq_name = seq_name.replace(".txt", "")
                        
                    all_sequence_names.append(seq_name)
        
        return sorted(list(set(all_sequence_names)))

    def _get_sequence_infos(self) -> dict:
        """
        Overrides the parent method.
        Manually constructs the sequence_infos dictionary.
        This must be called *before* the parent __init__.
        """
        all_sequence_names = self._get_all_sequence_names_from_splits()
        sequence_infos = dict()
        
        # P-DESTRE specifications (default)
        IMG_WIDTH = 3840
        IMG_HEIGHT = 2160
        FRAME_RATE = 30

        for sequence_name in all_sequence_names:
            # Annotation path
            gt_file_path = os.path.join(self.data_root, self.sub_dir, "annotations", f"{sequence_name}.txt")

            if not os.path.exists(gt_file_path):
                print(f"Warning: Annotation file not found for sequence {sequence_name} at {gt_file_path}, skipping.")
                continue

            # Count actual image files instead of relying on annotation frame IDs
            # This prevents mismatches where annotations reference frames that don't exist
            image_dir = os.path.join(self.data_root, self.sub_dir, "images", sequence_name, "img1")
            if not os.path.exists(image_dir):
                print(f"Warning: Image directory not found for sequence {sequence_name} at {image_dir}, skipping.")
                continue
            
            # Get sorted list of actual frame numbers from files (not just count)
            image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
            
            if len(image_files) == 0:
                print(f"Warning: No image files found for sequence {sequence_name} in {image_dir}, skipping.")
                continue
            
            # Extract frame numbers and check for gaps
            frame_numbers = []
            for img_file in image_files:
                try:
                    # Remove .jpg extension and convert to int
                    frame_num = int(img_file.replace('.jpg', ''))
                    frame_numbers.append(frame_num)
                except ValueError:
                    print(f"Warning: Invalid image filename '{img_file}' in {image_dir}, skipping file.")
                    continue
            
            if len(frame_numbers) == 0:
                print(f"Warning: No valid image files found for sequence {sequence_name}, skipping.")
                continue
            
            # Check if frames are consecutive
            expected_frames = set(range(1, max(frame_numbers) + 1))
            actual_frames = set(frame_numbers)
            missing_frames = expected_frames - actual_frames
            
            if missing_frames:
                print(f"Warning: Sequence {sequence_name} has {len(missing_frames)} missing frames (e.g., {sorted(list(missing_frames))[:5]}...)")
                print(f"         Using only the first consecutive segment ({frame_numbers[0]} to {frame_numbers[0] + len(frame_numbers) - 1}).")
                # Find the longest consecutive sequence starting from the beginning
                consecutive_count = 1
                for i in range(len(frame_numbers) - 1):
                    if frame_numbers[i+1] == frame_numbers[i] + 1:
                        consecutive_count += 1
                    else:
                        break
                max_frame = consecutive_count
            else:
                max_frame = len(frame_numbers)

            sequence_infos[sequence_name] = {
                "width": IMG_WIDTH,
                "height": IMG_HEIGHT,
                "length": max_frame,
                "imDir": "img1",      # Correct image directory
                "frameRate": FRAME_RATE,
                "imExt": ".jpg",
                "is_static": False  # P-DESTRE is a video dataset
            }
        return sequence_infos
    
    def get_image_path(self, data_root: str, split: str, sequence_name: str, frame_id: int):
        """
        Overrides the parent method to match the P-DESTRE file structure.
        Path: data_root/images/sequence_name/img1/frame_id.jpg
        Example: ./data/P-DESTRE/images/13-11-2019-1-1/img1/000001.jpg
        
        Args:
            data_root (str): The path from parent, e.g., "./data/P-DESTRE"
            split (str): The split name, e.g., "train". THIS IS IGNORED.
            sequence_name (str): The name of the sequence, e.g., "13-11-2019-1-1"
            frame_id (int): The 0-based frame index.
        """
        im_dir = self.sequence_infos[sequence_name]["imDir"]  # This will be "img1"
        im_ext = self.sequence_infos[sequence_name]["imExt"]  # This will be ".jpg"

        # P-DESTRE uses 6-digit padding and is 1-based (frame_id is 0-based index)
        image_name = f"{frame_id + 1:06d}{im_ext}" 

        # We IGNORE the 'split' argument as it's not in the P-DESTRE image path.
        # We use 'data_root' as the base, which is "./data/P-DESTRE".
        # We *ADD* the "images" folder.
        return os.path.join(
            data_root, "images", sequence_name, im_dir, image_name
        )

    def _get_image_paths(self):
        """
        Overrides the parent method to match the P-DESTRE file structure.
        P-DESTRE path: data_dir/images/sequence_name/img1/NNNNNN.jpg
        (no train/test subdirectory, 6-digit frame numbers)
        """
        sequence_names = self._get_sequence_names()
        image_paths = defaultdict(list)
        for sequence_name in sequence_names:
            im_dir = self.sequence_infos[sequence_name]["imDir"]  # "img1"
            im_ext = self.sequence_infos[sequence_name]["imExt"]  # ".jpg"
            for i in range(self.sequence_infos[sequence_name]["length"]):
                # P-DESTRE uses 6-digit padding, 1-based frame numbers
                image_name = f"{i + 1:06d}{im_ext}"
                # Path: data_dir/images/sequence_name/img1/NNNNNN.jpg
                image_path = os.path.join(
                    self.data_dir, "images", sequence_name, im_dir, image_name
                )
                image_paths[sequence_name].append(image_path)
        return image_paths

    def _get_sequence_names(self) -> list:
        """
        Overrides the parent method.
        Gets sequence names from the already-populated self.sequence_infos.
        """
        # This ensures we only return sequences that had valid .txt files
        return sorted(list(self.sequence_infos.keys()))
    
    def _init_annotations(self, sequence_names: list) -> dict:
        """
        Initialize the annotation dictionary structure, adding the 'concepts' field.
        Now supports 7 concepts stored as a 2D tensor (N_objects, N_concepts).
        Concepts: gender, hairstyle, head_accessories, upper_body, lower_body, feet, accessories
        """
        annotations = defaultdict(lambda: defaultdict(list))
        for sequence_name in sequence_names:
            if sequence_name not in self.sequence_infos:
                continue
            length = self.sequence_infos[sequence_name]["length"]
            for i in range(length):
                annotations[sequence_name][i] = {
                    "id": torch.empty(size=(0,), dtype=torch.int64),
                    "category": torch.empty(size=(0,), dtype=torch.int64),
                    "bbox": torch.empty(size=(0, 4), dtype=torch.float32),
                    "visibility": torch.empty(size=(0,), dtype=torch.float32),
                    # 2D tensor for 7 concepts: (N_objects, 7)
                    "concepts": torch.empty(size=(0, 7), dtype=torch.int64), 
                }
        return annotations

    def _get_annotations(self) -> dict:
        """
        Load and parse P-DESTRE annotations.
        """
        sequence_names = self._get_sequence_names()
        annotations = self._init_annotations(sequence_names)
        
        # Use self.data_dir (from parent)
        data_dir = self.data_dir 

        for sequence_name in sequence_names:
            gt_file_path = os.path.join(data_dir, "annotations", f"{sequence_name}.txt")

            if not os.path.exists(gt_file_path):
                continue

            with open(gt_file_path, "r") as f:
                for line in f.readlines():
                    try:
                        items = line.strip().split(',')
                        # P-DESTRE annotation columns:
                        # Frame,ID,x,y,h,w,conf,world_x,world_y,world_z,
                        # Gender(10), ..., Hairstyle(16), ..., Head_Accessories(20), Upper_Body(21), Lower_Body(22), Feet(23), Accessories(24)
                        if len(items) < 25:  # Need at least 25 columns for all concepts
                            continue
                        
                        obj_id = int(items[1])
                        
                        # Ignore "distractor" / "background" detections
                        if obj_id == -1:
                            continue

                        frame_id = int(items[0])
                        x = float(items[2])  # bb_left
                        y = float(items[3])  # bb_top
                        h = float(items[4])  # height (P-DESTRE format)
                        w = float(items[5])  # width (P-DESTRE format)
                        
                        # Parse all 7 concepts
                        gender = int(items[10])              # 0=Male, 1=Female, 2=Unknown
                        hairstyle = int(items[16])           # 0=Bald, 1=Short, 2=Medium, 3=Long, 4=Horse Tail, 5=Unknown
                        head_accessories = int(items[20])    # 0=Hat, 1=Scarf, 2=Neckless, 3=Cannot see, 4=Unknown
                        upper_body = int(items[21])          # 0-12 clothing types
                        lower_body = int(items[22])          # 0=Jeans, 1=Leggins, ..., 9=Unknown
                        feet = int(items[23])                # 0=Sport Shoe, ..., 6=Unknown
                        accessories = int(items[24])         # 0=Bag, ..., 7=Unknown
                        
                        # P-DESTRE uses [x, y, h, w] but MOT format needs [x, y, w, h]
                        # So we swap h and w when creating the bbox
                        bbox = [x, y, h, w]
                        
                        category = 0
                        visibility = 1.0
                        # Store all 7 concepts as a list
                        concepts_data = [gender, hairstyle, head_accessories, upper_body, lower_body, feet, accessories]

                        ann_index = frame_id - 1  # P-DESTRE is 1-indexed

                        if ann_index < 0 or ann_index >= len(annotations[sequence_name]):
                            continue
                            
                        append_annotation(
                            annotations[sequence_name][ann_index],
                            obj_id,
                            category,
                            bbox,
                            visibility,
                            concepts_data
                        )
                    except (ValueError, IndexError) as e:
                        # print(f"Skipping bad line in {sequence_name}: {line.strip()} | Error: {e}")
                        continue

        # Determine annotation legality
        for sequence_name in sequence_names:
            for i in range(self.sequence_infos[sequence_name]["length"]):
                annotations[sequence_name][i]["is_legal"] = is_legal(annotations[sequence_name][i])

        return annotations