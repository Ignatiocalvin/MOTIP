# CLIP-based concept classifier for comparison with MOTIP predictions
# This module provides CLIP predictions to compare against ground truth and MOTIP

import torch
import clip
from PIL import Image
import numpy as np
from typing import List, Tuple, Dict, Optional


class CLIPConceptClassifier:
    """
    CLIP-based concept classifier for gender and clothing classification.
    Used to compare CLIP predictions against MOTIP predictions and ground truth.
    """
    
    def __init__(self, model_name: str = "ViT-B/32", device: str = None):
        """
        Initialize CLIP classifier.
        
        Args:
            model_name: CLIP model variant (e.g., "ViT-B/32", "ViT-L/14")
            device: Device to run on (cuda/cpu)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = None
        self.preprocess = None
        self.text_embeddings = {}
        self._initialized = False
        
        # Define concept prompts
        self.concept_prompts = {
            'gender': {
                0: ["a photo of a man", "a male person", "a photo of a boy"],  # Male
                1: ["a photo of a woman", "a female person", "a photo of a girl"],  # Female
                2: ["a person with unclear gender", "an androgynous person"],  # Unknown
            },
            'upper_body': {
                0: ["a person wearing a t-shirt", "someone in a t-shirt"],  # T-Shirt
                1: ["a person wearing a blouse", "someone in a blouse"],  # Blouse
                2: ["a person wearing a sweater", "someone in a sweater"],  # Sweater
                3: ["a person wearing a coat", "someone in a coat or jacket"],  # Coat
                4: ["a person wearing a bikini top", "someone in swimwear"],  # Bikini
                5: ["a shirtless person", "someone with no shirt"],  # Naked
                6: ["a person wearing a dress", "someone in a dress"],  # Dress
                7: ["a person in uniform", "someone wearing a uniform"],  # Uniform
                8: ["a person wearing a shirt", "someone in a button-up shirt"],  # Shirt
                9: ["a person wearing a suit", "someone in business attire"],  # Suit
                10: ["a person wearing a hoodie", "someone in a hooded sweatshirt"],  # Hoodie
                11: ["a person wearing a cardigan", "someone in a cardigan"],  # Cardigan
                12: ["a person with unclear clothing", "someone with obscured top"],  # Unknown
            }
        }
        
        # Concept class names for logging
        self.concept_names = {
            'gender': {0: 'Male', 1: 'Female', 2: 'Unknown'},
            'upper_body': {
                0: 'T-Shirt', 1: 'Blouse', 2: 'Sweater', 3: 'Coat',
                4: 'Bikini', 5: 'Naked', 6: 'Dress', 7: 'Uniform',
                8: 'Shirt', 9: 'Suit', 10: 'Hoodie', 11: 'Cardigan', 12: 'Unknown'
            }
        }
    
    def initialize(self):
        """Load CLIP model and precompute text embeddings."""
        if self._initialized:
            return
        
        try:
            print(f"[CLIP] Loading CLIP model: {self.model_name}...")
            self.model, self.preprocess = clip.load(self.model_name, device=self.device)
            self.model.eval()
            
            # Precompute text embeddings for all concepts
            print("[CLIP] Precomputing text embeddings...")
            with torch.no_grad():
                for concept_name, prompts_dict in self.concept_prompts.items():
                    self.text_embeddings[concept_name] = {}
                    for class_id, prompts in prompts_dict.items():
                        embeddings = []
                        for prompt in prompts:
                            text_tokens = clip.tokenize([prompt]).to(self.device)
                            text_embedding = self.model.encode_text(text_tokens)
                            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
                            embeddings.append(text_embedding)
                        # Average embeddings for each class
                        self.text_embeddings[concept_name][class_id] = torch.mean(
                            torch.cat(embeddings, dim=0), dim=0, keepdim=True
                        )
            
            self._initialized = True
            print(f"[CLIP] Initialized successfully on {self.device}")
            
        except Exception as e:
            print(f"[CLIP] Failed to initialize: {e}")
            self._initialized = False
    
    def classify_crop(self, crop_tensor: torch.Tensor, concept: str = 'gender') -> Tuple[int, float]:
        """
        Classify a single image crop.
        
        Args:
            crop_tensor: Image tensor of shape (C, H, W) or (1, C, H, W), already preprocessed
            concept: Concept to classify ('gender' or 'upper_body')
            
        Returns:
            Tuple of (predicted_class, confidence)
        """
        if not self._initialized:
            return (2, 0.0)  # Unknown with 0 confidence
        
        if concept not in self.text_embeddings:
            return (2, 0.0)
        
        with torch.no_grad():
            # Ensure correct shape
            if crop_tensor.dim() == 3:
                crop_tensor = crop_tensor.unsqueeze(0)
            
            crop_tensor = crop_tensor.to(self.device)
            
            # Get image features
            image_features = self.model.encode_image(crop_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Compute similarities with each class
            similarities = {}
            for class_id, text_embedding in self.text_embeddings[concept].items():
                similarity = torch.cosine_similarity(image_features, text_embedding, dim=-1)
                similarities[class_id] = similarity.item()
            
            # Get best prediction
            predicted_class = max(similarities.keys(), key=lambda k: similarities[k])
            confidence = similarities[predicted_class]
            
            return (predicted_class, confidence)
    
    def classify_batch_from_image(
        self, 
        image_tensor: torch.Tensor,
        bboxes: List[List[float]],
        concept: str = 'gender'
    ) -> List[Tuple[int, float]]:
        """
        Classify multiple bounding box crops from a single image.
        
        Args:
            image_tensor: Full image tensor of shape (C, H, W) or (B, C, H, W)
            bboxes: List of bounding boxes [[x1, y1, x2, y2], ...]
            concept: Concept to classify
            
        Returns:
            List of (predicted_class, confidence) tuples
        """
        if not self._initialized:
            return [(2, 0.0)] * len(bboxes)
        
        predictions = []
        
        # Handle batch dimension
        if image_tensor.dim() == 4:
            image_tensor = image_tensor[0]  # Take first image
        
        _, h, w = image_tensor.shape
        
        for bbox in bboxes:
            x1, y1, x2, y2 = [int(b) for b in bbox]
            
            # Clamp to image boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                predictions.append((2, 0.0))  # Unknown
                continue
            
            # Extract crop
            crop = image_tensor[:, y1:y2, x1:x2]
            
            # Resize to CLIP input size (224x224)
            crop = torch.nn.functional.interpolate(
                crop.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
            ).squeeze(0)
            
            # Classify
            pred_class, confidence = self.classify_crop(crop, concept)
            predictions.append((pred_class, confidence))
        
        return predictions
    
    def get_predictions_for_logging(
        self,
        image_tensor: torch.Tensor,
        bboxes: List[List[float]],
        gt_gender: Optional[List[int]] = None,
        gt_upper_body: Optional[List[int]] = None,
        motip_gender: Optional[List[int]] = None,
        motip_upper_body: Optional[List[int]] = None,
    ) -> Dict:
        """
        Get CLIP predictions for logging and comparison.
        
        Args:
            image_tensor: Image tensor
            bboxes: List of bounding boxes
            gt_gender: Ground truth gender labels
            gt_upper_body: Ground truth upper body clothing labels
            motip_gender: MOTIP predicted gender labels
            motip_upper_body: MOTIP predicted upper body labels
            
        Returns:
            Dictionary with predictions and comparison metrics
        """
        if not self._initialized or len(bboxes) == 0:
            return {}
        
        results = {
            'n_objects': len(bboxes),
            'clip_gender': [],
            'clip_upper_body': [],
            'clip_gender_conf': [],
            'clip_upper_body_conf': [],
        }
        
        # Get CLIP predictions
        gender_preds = self.classify_batch_from_image(image_tensor, bboxes, 'gender')
        upper_body_preds = self.classify_batch_from_image(image_tensor, bboxes, 'upper_body')
        
        for (g_pred, g_conf), (ub_pred, ub_conf) in zip(gender_preds, upper_body_preds):
            results['clip_gender'].append(g_pred)
            results['clip_upper_body'].append(ub_pred)
            results['clip_gender_conf'].append(g_conf)
            results['clip_upper_body_conf'].append(ub_conf)
        
        # Calculate accuracy vs ground truth
        if gt_gender is not None:
            gt_gender_matches = sum(1 for c, g in zip(results['clip_gender'], gt_gender) if c == g)
            results['clip_vs_gt_gender_acc'] = gt_gender_matches / len(gt_gender) * 100 if gt_gender else 0
        
        if gt_upper_body is not None:
            gt_ub_matches = sum(1 for c, g in zip(results['clip_upper_body'], gt_upper_body) if c == g)
            results['clip_vs_gt_upper_body_acc'] = gt_ub_matches / len(gt_upper_body) * 100 if gt_upper_body else 0
        
        # Calculate accuracy vs MOTIP
        if motip_gender is not None:
            motip_gender_matches = sum(1 for c, m in zip(results['clip_gender'], motip_gender) if c == m)
            results['clip_vs_motip_gender_acc'] = motip_gender_matches / len(motip_gender) * 100 if motip_gender else 0
        
        if motip_upper_body is not None:
            motip_ub_matches = sum(1 for c, m in zip(results['clip_upper_body'], motip_upper_body) if c == m)
            results['clip_vs_motip_upper_body_acc'] = motip_ub_matches / len(motip_upper_body) * 100 if motip_upper_body else 0
        
        return results
    
    def format_comparison_log(
        self,
        gt_gender: List[int],
        gt_upper_body: List[int],
        motip_gender: List[int],
        motip_upper_body: List[int],
        clip_gender: List[int],
        clip_upper_body: List[int],
        max_samples: int = 15
    ) -> str:
        """
        Format a comparison log string showing GT vs MOTIP vs CLIP predictions.
        
        Args:
            gt_gender: Ground truth gender labels
            gt_upper_body: Ground truth upper body labels
            motip_gender: MOTIP predictions for gender
            motip_upper_body: MOTIP predictions for upper body
            clip_gender: CLIP predictions for gender
            clip_upper_body: CLIP predictions for upper body
            max_samples: Maximum samples to show
            
        Returns:
            Formatted log string
        """
        n = min(max_samples, len(gt_gender))
        
        lines = []
        lines.append(f"\n{'='*80}")
        lines.append("CONCEPT PREDICTIONS COMPARISON (GT vs MOTIP vs CLIP)")
        lines.append(f"{'='*80}")
        
        # Gender comparison
        lines.append(f"\n[GENDER] (0=Male, 1=Female, 2=Unknown)")
        gt_str = ','.join([str(g) for g in gt_gender[:n]])
        motip_str = ','.join([str(m) for m in motip_gender[:n]])
        clip_str = ','.join([str(c) for c in clip_gender[:n]])
        
        lines.append(f"  GT:    [{gt_str}]")
        lines.append(f"  MOTIP: [{motip_str}]")
        lines.append(f"  CLIP:  [{clip_str}]")
        
        # Calculate accuracies
        motip_gender_acc = sum(1 for g, m in zip(gt_gender, motip_gender) if g == m) / len(gt_gender) * 100 if gt_gender else 0
        clip_gender_acc = sum(1 for g, c in zip(gt_gender, clip_gender) if g == c) / len(gt_gender) * 100 if gt_gender else 0
        
        lines.append(f"  MOTIP vs GT: {motip_gender_acc:.1f}% | CLIP vs GT: {clip_gender_acc:.1f}%")
        
        # Upper body comparison
        lines.append(f"\n[UPPER_BODY] (0=T-Shirt, 1=Blouse, 2=Sweater, 3=Coat, ...)")
        gt_str = ','.join([str(g) for g in gt_upper_body[:n]])
        motip_str = ','.join([str(m) for m in motip_upper_body[:n]])
        clip_str = ','.join([str(c) for c in clip_upper_body[:n]])
        
        lines.append(f"  GT:    [{gt_str}]")
        lines.append(f"  MOTIP: [{motip_str}]")
        lines.append(f"  CLIP:  [{clip_str}]")
        
        # Calculate accuracies
        motip_ub_acc = sum(1 for g, m in zip(gt_upper_body, motip_upper_body) if g == m) / len(gt_upper_body) * 100 if gt_upper_body else 0
        clip_ub_acc = sum(1 for g, c in zip(gt_upper_body, clip_upper_body) if g == c) / len(gt_upper_body) * 100 if gt_upper_body else 0
        
        lines.append(f"  MOTIP vs GT: {motip_ub_acc:.1f}% | CLIP vs GT: {clip_ub_acc:.1f}%")
        
        lines.append(f"{'='*80}")
        
        return '\n'.join(lines)


# Global CLIP classifier instance (lazy initialization)
_clip_classifier: Optional[CLIPConceptClassifier] = None


def get_clip_classifier(device: str = None) -> CLIPConceptClassifier:
    """Get or create the global CLIP classifier instance."""
    global _clip_classifier
    if _clip_classifier is None:
        _clip_classifier = CLIPConceptClassifier(device=device)
    return _clip_classifier
