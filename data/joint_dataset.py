# Copyright (c) Ruopeng Gao. All Rights Reserved.
# No matter how many datasets you used in your code,
# you should always use JointDataset to combine and organize them (even if you only used one dataset).

import copy
import os # <-- Import OS
import torch
from collections import defaultdict
from torch.utils.data import Dataset
from PIL import Image

from .dancetrack import DanceTrack
from .sportsmot import SportsMOT
from .crowdhuman import CrowdHuman
from .bft import BFT
from .pdestre import PDESTRE


dataset_classes = {
    "DanceTrack": DanceTrack,
    "SportsMOT": SportsMOT,
    "CrowdHuman": CrowdHuman,
    "BFT": BFT,
    "P-DESTRE": PDESTRE, # <-- This MUST match the string in your config
}


class JointDataset(Dataset):
    def __init__(
            self,
            data_root: str,
            datasets: list,
            splits: list,
            transforms=None,
            **kwargs,
    ):
        """
        Args:
            data_root: The root directory of datasets.
            datasets: The list of dataset names, e.g., ["DanceTrack", "SportsMOT"].
            splits: The list of (dataset) split names, e.g., ["train", "train"].
        """
        super().__init__()
        assert len(datasets) == len(splits), "The number of datasets and splits should be the same."
        self.transforms = transforms

        # Handle the parameters **kwargs:
        self.size_divisibility = kwargs.get("size_divisibility", 0)

        # Load the datasets into "sequence_infos", "image_paths", and "annotations",
        # each of which is a dictionary with the dataset name and split as the key.
        # e.g., sequence_infos["DanceTrack"]["train"]["sequence_name"] = {}.
        self.sequence_infos = defaultdict(lambda: defaultdict(dict))
        self.image_paths = defaultdict(lambda: defaultdict(dict))
        self.annotations = defaultdict(lambda: defaultdict(dict))
        for dataset, split in zip(datasets, splits):
            try:
                if dataset not in dataset_classes:
                    raise KeyError(f"Dataset '{dataset}' not found in dataset_classes.")
                
                dataset_class = dataset_classes[dataset](
                    data_root=data_root,
                    split=split,
                    load_annotation=True,
                )
                self.sequence_infos[dataset][split] = dataset_class.get_sequence_infos()
                self.image_paths[dataset][split] = dataset_class.get_image_paths()
                self.annotations[dataset][split] = dataset_class.get_annotations()

                # --- BEGIN HOTFIX for P-DESTRE Pathing Issue ---
                # This block will run *after* the (broken) paths are loaded
                # and will manually overwrite them with the correct path structure.
                if dataset == "P-DESTRE":
                    print(f"[HOTFIX] Applying path correction for P-DESTRE split: {split}")
                    
                    # This will hold the new, correct paths
                    corrected_paths_dict = defaultdict(lambda: defaultdict(dict))
                    
                    # This is the base path, e.g., "./data/P-DESTRE"
                    pdestre_base_path = os.path.join(data_root, dataset_class.sub_dir)
                    
                    # Iterate over the sequences we've already loaded
                    for seq_name, seq_info in self.sequence_infos[dataset][split].items():
                        im_dir = seq_info.get("imDir", "img1")
                        im_ext = seq_info.get("imExt", ".jpg")
                        
                        for frame_idx in range(seq_info["length"]):
                            # P-DESTRE uses 6-digit padding and is 1-based
                            image_name = f"{frame_idx + 1:06d}{im_ext}"
                            
                            # Build the correct path:
                            # e.g., ./data/P-DESTRE/images/11-11-2019-1-6/img1/000001.jpg
                            correct_path = os.path.join(
                                pdestre_base_path, "images", seq_name, im_dir, image_name
                            )
                            corrected_paths_dict[seq_name][frame_idx] = correct_path
                    
                    # Overwrite the broken paths in self.image_paths
                    self.image_paths[dataset][split] = corrected_paths_dict
                    print(f"[HOTFIX] Path correction applied.")
                # --- END HOTFIX ---

            except KeyError:
                raise AttributeError(f"Dataset {dataset} is not supported.")
        # Decouple the 'is_legal' attribute from the annotations,
        # I believe it is more flexible to check the legality of the annotations in the sampling process.
        self.ann_is_legals = self._decouple_is_legal()

        # Init the sampling details:
        # Here, they are not ready for sampling,
        # you should call "self.set_sample_details()" to prepare them.
        self.sample_begins: list | None = None      # a tuple: (dataset, split, sequence_name, begin_index)
        return

    def _decouple_is_legal(self):
        decoupled_is_legal = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for dataset in self.annotations:
            for split in self.annotations[dataset]:
                for sequence_name in self.annotations[dataset][split]:
                    # The object self.annotations[dataset][split][sequence_name] is a
                    # dictionary mapping frame_id (int) to an annotation dict, NOT a list.
                    # We must iterate by frame_id from 0 to the sequence length.

                    # Get the sequence length from sequence_infos
                    try:
                        seq_length = self.sequence_infos[dataset][split][sequence_name]["length"]
                    except KeyError:
                        print(f"Warning: Could not find sequence info for {dataset}/{split}/{sequence_name} in _decouple_is_legal. Skipping.")
                        continue
                    
                    seq_ann_dict = self.annotations[dataset][split][sequence_name]

                    for frame_id in range(seq_length):
                        try:
                            # Access the annotation dict for this specific frame
                            annotation = seq_ann_dict[frame_id]
                            decoupled_is_legal[dataset][split][sequence_name].append(annotation["is_legal"])
                        except KeyError:
                            # This can happen if _init_annotations or _get_annotations failed
                            # to create an entry for this frame_id.
                            print(f"Warning: Missing annotation entry for {dataset}/{split}/{sequence_name} frame {frame_id}. Appending is_legal=False.")
                            decoupled_is_legal[dataset][split][sequence_name].append(False)
                        except TypeError:
                             # This can happen if seq_ann_dict[frame_id] is not a dict (e.g., still a defaultdict)
                            print(f"Warning: Invalid annotation entry for {dataset}/{split}/{sequence_name} frame {frame_id}. Appending is_legal=False.")
                            decoupled_is_legal[dataset][split][sequence_name].append(False)
        # Reformat the 'is_legal' attribute from a list to a tensor,
        # which is more convenient for the sampling process (calculation-friendly).
        decoupled_is_legal_in_tensor = defaultdict(lambda: defaultdict(lambda: defaultdict(torch.Tensor)))
        for dataset in decoupled_is_legal:
            for split in decoupled_is_legal[dataset]:
                for sequence_name in decoupled_is_legal[dataset][split]:
                    decoupled_is_legal_in_tensor[dataset][split][sequence_name] = torch.tensor(
                        decoupled_is_legal[dataset][split][sequence_name], dtype=torch.bool
                    )
        return decoupled_is_legal_in_tensor

    def set_sample_details(
            self,
            sample_length: int,
            sample_interval: int,
            sample_mode: str = "random_interval",
    ):
        """
        Set the details for sampling.
        Now we only have "self.sample_begins" to store the beginning of each legal sample.
        NOTE: You should call this function at the start of each epoch.
        Args:
            sample_length: The length of each sample.
            sample_interval: The interval between two adjacent samples, currently not used.
            sample_mode: The mode of sampling, e.g., "random_interval", "fixed_interval".
        """
        assert sample_mode in ["random_interval"], f"Sample mode '{sample_mode}' is not supported."
        self.sample_begins = list()
        for dataset in self.annotations:
            for split in self.annotations[dataset]:
                for sequence_name in self.annotations[dataset][split]:
                    for frame_id in range(self.sequence_infos[dataset][split][sequence_name]["length"]):
                        if self.sequence_infos[dataset][split][sequence_name]["is_static"] is True:     # static image
                            self.sample_begins.append((dataset, split, sequence_name, frame_id))
                        else:   # real-world video
                            if frame_id + sample_length <= self.sequence_infos[dataset][split][sequence_name]["length"]:
                                if self.ann_is_legals[dataset][split][sequence_name][frame_id: frame_id + sample_length].all():
                                    # TODO: We may support different sampling ratio for each dataset, need to add code.
                                    self.sample_begins.append((dataset, split, sequence_name, frame_id))
        return

    def __len__(self):
        assert self.sample_begins is not None, "Please use 'self.set_sample_details()' at the start of each epoch."
        return len(self.sample_begins)

    def __getitem__(self, info):
        dataset = info["dataset"]
        split = info["split"]
        sequence = info["sequence"]
        frame_idxs = info["frame_idxs"]
        # Get image paths:
        image_paths = [
            self.image_paths[dataset][split][sequence][frame_idx] for frame_idx in frame_idxs
        ]
        # Read images and filter out missing ones:
        images = []
        valid_frame_idxs = []
        for frame_idx, image_path in zip(frame_idxs, image_paths):
            try:
                images.append(Image.open(image_path))
                valid_frame_idxs.append(frame_idx)
            except FileNotFoundError:
                # Skip missing images silently
                continue
        
        # If all images are missing, skip this sample by returning the next one
        if len(images) == 0:
            return self.__getitem__((idx + 1) % len(self))
        
        # Update frame_idxs to only include valid frames
        frame_idxs = valid_frame_idxs
        
        # Get annotations:
        annotations = [
            self.annotations[dataset][split][sequence][frame_idx] for frame_idx in frame_idxs
        ]   # "bbox", "category", "id", "visibility", "is_legal"
        # Get metas:
        metas = [
            {
                "dataset": dataset,
                "split": split,
                "sequence": sequence,
                "frame_idx": frame_idx,
                "is_static": self.sequence_infos[dataset][split][sequence]["is_static"],
                "is_begin": False,      # whether the frame is the beginning of a video clip
                "size_divisibility": self.size_divisibility,
            } for frame_idx in frame_idxs
        ]
        # Do some modifications:
        metas[0]["is_begin"] = True     # the first frame is the beginning of a video clip
        # Deep copy:
        annotations = [copy.deepcopy(annotation) for annotation in annotations]
        metas = [copy.deepcopy(meta) for meta in metas]

        # Apply transforms:
        if self.transforms is not None:
            images, annotations, metas = self.transforms(images, annotations, metas)
        # from .tools import visualize_a_batch
        # visualize_a_batch(images, annotations)
        return images, annotations, metas

    def statistics(self):
        """
        Return the statistics of the dataset, in a list.
        Each item is a string: "Dancetrack.train, 35 sequences, 40000 frames."
        """
        statistics = list()
        for dataset in self.sequence_infos:
            for split in self.sequence_infos[dataset]:
                num_sequences = len(self.sequence_infos[dataset][split])
                num_frames = sum([info["length"] for info in self.sequence_infos[dataset][split].values()])
                statistics.append(f"{dataset}.{split}, {num_sequences} sequences, {num_frames} frames.")
        return statistics