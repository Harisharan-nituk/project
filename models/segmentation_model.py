import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import os
import requests

logger = logging.getLogger(__name__)

class SegmentationModel:
    """Image segmentation model for human parsing and object detection"""
    
    def __init__(self, model_name: str = 'deeplabv3_resnet50', device: str = 'cpu'):
        """
        Initialize segmentation model
        
        Args:
            model_name: Name of the segmentation model
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_loaded = False
        
        # Segmentation classes (COCO dataset)
        self.coco_classes = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
            'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A',
            'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        # Human parsing classes (for detailed body part segmentation)
        self.human_parsing_classes = [
            'background', 'hat', 'hair', 'glove', 'sunglasses', 'upper-clothes',
            'dress', 'coat', 'socks', 'pants', 'jumpsuits', 'scarf', 'skirt',
            'face', 'left-arm', 'right-arm', 'left-leg', 'right-leg', 'left-shoe', 'right-shoe'
        ]
        
        # Color map for visualization
        self.color_map = self._create_color_map()
        
        # Preprocessing transforms
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize model
        self._load_model()
    
    def _load_model(self):
        """Load the segmentation model"""
        try:
            logger.info(f"Loading segmentation model: {self.model_name}")
            
            if self.model_name == 'deeplabv3_resnet50':
                self.model = models.segmentation.deeplabv3_resnet50(pretrained=True)
            elif self.model_name == 'deeplabv3_resnet101':
                self.model = models.segmentation.deeplabv3_resnet101(pretrained=True)
            elif self.model_name == 'fcn_resnet50':
                self.model = models.segmentation.fcn_resnet50(pretrained=True)
            elif self.model_name == 'fcn_resnet101':
                self.model = models.segmentation.fcn_resnet101(pretrained=True)
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")
            
            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            
            logger.info(f"Segmentation model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading segmentation model: {str(e)}")
            raise
    
    def segment_image(self, image: np.ndarray, confidence_threshold: float = 0.7) -> Dict:
        """
        Perform semantic segmentation on image
        
        Args:
            image: Input image
            confidence_threshold: Confidence threshold for segmentation
            
        Returns:
            Dictionary containing segmentation results
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        try:
            # Preprocess image
            original_height, original_width = image.shape[:2]
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize for model input
            input_image = cv2.resize(rgb_image, (512, 512))
            
            # Convert to tensor
            input_tensor = self.preprocess(input_image).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                output = self.model(input_tensor)['out'][0]
                output_predictions = output.argmax(0).cpu().numpy()
            
            # Resize back to original size
            segmentation_mask = cv2.resize(
                output_predictions.astype(np.uint8), 
                (original_width, original_height), 
                interpolation=cv2.INTER_NEAREST
            )
            
            # Get confidence scores
            confidence_scores = torch.nn.functional.softmax(output, dim=0).cpu().numpy()
            max_confidence = np.max(confidence_scores, axis=0)
            max_confidence = cv2.resize(max_confidence, (original_width, original_height))
            
            # Apply confidence threshold
            low_confidence_mask = max_confidence < confidence_threshold
            segmentation_mask[low_confidence_mask] = 0  # Set to background
            
            # Create results dictionary
            results = {
                'segmentation_mask': segmentation_mask,
                'confidence_scores': max_confidence,
                'class_masks': self._create_class_masks(segmentation_mask),
                'person_mask': self._extract_person_mask(segmentation_mask),
                'object_counts': self._count_objects(segmentation_mask)
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in image segmentation: {str(e)}")
            return {}
    
    def segment_person(self, image: np.ndarray) -> np.ndarray:
        """
        Segment person from image
        
        Args:
            image: Input image
            
        Returns:
            Person segmentation mask
        """
        segmentation_results = self.segment_image(image)
        
        if 'person_mask' in segmentation_results:
            return segmentation_results['person_mask']
        else:
            return np.zeros(image.shape[:2], dtype=np.uint8)
    
    def _extract_person_mask(self, segmentation_mask: np.ndarray) -> np.ndarray:
        """Extract mask for person class"""
        person_class_id = 1  # Person class in COCO
        person_mask = (segmentation_mask == person_class_id).astype(np.uint8) * 255
        
        # Apply morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_OPEN, kernel)
        person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_CLOSE, kernel)
        
        return person_mask
    
    def _create_class_masks(self, segmentation_mask: np.ndarray) -> Dict[str, np.ndarray]:
        """Create individual masks for each detected class"""
        class_masks = {}
        
        unique_classes = np.unique(segmentation_mask)
        
        for class_id in unique_classes:
            if class_id < len(self.coco_classes):
                class_name = self.coco_classes[class_id]
                if class_name != 'N/A':
                    mask = (segmentation_mask == class_id).astype(np.uint8) * 255
                    class_masks[class_name] = mask
        
        return class_masks
    
    def _count_objects(self, segmentation_mask: np.ndarray) -> Dict[str, int]:
        """Count objects in segmentation mask"""
        object_counts = {}
        
        unique_classes, counts = np.unique(segmentation_mask, return_counts=True)
        
        for class_id, count in zip(unique_classes, counts):
            if class_id < len(self.coco_classes) and class_id > 0:  # Skip background
                class_name = self.coco_classes[class_id]
                if class_name != 'N/A':
                    object_counts[class_name] = int(count)
        
        return object_counts
    
    def _create_color_map(self) -> np.ndarray:
        """Create color map for visualization"""
        num_classes = len(self.coco_classes)
        color_map = np.zeros((num_classes, 3), dtype=np.uint8)
        
        # Generate distinct colors for each class
        for i in range(num_classes):
            color_map[i] = [
                (i * 67) % 256,
                (i * 113) % 256,
                (i * 197) % 256
            ]
        
        return color_map
    
    def visualize_segmentation(self, image: np.ndarray, segmentation_mask: np.ndarray, 
                              alpha: float = 0.5) -> np.ndarray:
        """
        Visualize segmentation results
        
        Args:
            image: Original image
            segmentation_mask: Segmentation mask
            alpha: Transparency for overlay
            
        Returns:
            Visualization image
        """
        try:
            # Create colored segmentation
            colored_mask = self.color_map[segmentation_mask]
            
            # Blend with original image
            visualization = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
            
            return visualization
            
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            return image
    
    def extract_clothing_regions(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract clothing regions from image
        
        Args:
            image: Input image
            
        Returns:
            Dictionary of clothing region masks
        """
        segmentation_results = self.segment_image(image)
        
        if not segmentation_results:
            return {}
        
        class_masks = segmentation_results.get('class_masks', {})
        
        # Define clothing-related classes
        clothing_classes = ['person']  # We'll refine person mask for clothing
        
        clothing_regions = {}
        
        # For person class, we need to do additional processing
        if 'person' in class_masks:
            person_mask = class_masks['person']
            
            # Estimate clothing regions based on person segmentation
            clothing_regions.update(self._estimate_clothing_regions(image, person_mask))
        
        return clothing_regions
    
    def _estimate_clothing_regions(self, image: np.ndarray, person_mask: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Estimate clothing regions from person mask
        
        Args:
            image: Original image
            person_mask: Person segmentation mask
            
        Returns:
            Dictionary of clothing region masks
        """
        clothing_regions = {}
        
        try:
            # Find person contour
            contours, _ = cv2.findContours(person_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return clothing_regions
            
            # Get largest contour (main person)
            main_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(main_contour)
            
            # Estimate body regions based on proportions
            # Upper body (chest/torso area)
            upper_y_start = y + int(h * 0.15)  # Below head
            upper_y_end = y + int(h * 0.6)     # Above waist
            upper_mask = np.zeros_like(person_mask)
            upper_mask[upper_y_start:upper_y_end, x:x+w] = person_mask[upper_y_start:upper_y_end, x:x+w]
            clothing_regions['upper_body'] = upper_mask
            
            # Lower body (pants/skirt area)
            lower_y_start = y + int(h * 0.5)   # Around waist
            lower_y_end = y + int(h * 0.9)     # Above feet
            lower_mask = np.zeros_like(person_mask)
            lower_mask[lower_y_start:lower_y_end, x:x+w] = person_mask[lower_y_start:lower_y_end, x:x+w]
            clothing_regions['lower_body'] = lower_mask
            
            # Arms region
            arm_width = int(w * 0.3)
            
            # Left arm
            left_arm_mask = np.zeros_like(person_mask)
            left_arm_mask[upper_y_start:upper_y_end, x:x+arm_width] = \
                person_mask[upper_y_start:upper_y_end, x:x+arm_width]
            clothing_regions['left_arm'] = left_arm_mask
            
            # Right arm
            right_arm_mask = np.zeros_like(person_mask)
            right_arm_mask[upper_y_start:upper_y_end, x+w-arm_width:x+w] = \
                person_mask[upper_y_start:upper_y_end, x+w-arm_width:x+w]
            clothing_regions['right_arm'] = right_arm_mask
            
            # Feet region
            feet_y_start = y + int(h * 0.9)
            feet_mask = np.zeros_like(person_mask)
            feet_mask[feet_y_start:y+h, x:x+w] = person_mask[feet_y_start:y+h, x:x+w]
            clothing_regions['feet'] = feet_mask
            
        except Exception as e:
            logger.error(f"Error estimating clothing regions: {str(e)}")
        
        return clothing_regions
    
    def refine_person_mask(self, image: np.ndarray, initial_mask: np.ndarray) -> np.ndarray:
        """
        Refine person segmentation mask using additional techniques
        
        Args:
            image: Original image
            initial_mask: Initial segmentation mask
            
        Returns:
            Refined person mask
        """
        try:
            # Apply GrabCut for refinement
            mask = np.where(initial_mask > 0, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
            
            # Find bounding rectangle
            contours, _ = cv2.findContours(initial_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                main_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(main_contour)
                rect = (x, y, w, h)
                
                # Initialize models for GrabCut
                bgd_model = np.zeros((1, 65), np.float64)
                fgd_model = np.zeros((1, 65), np.float64)
                
                # Apply GrabCut
                cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
                
                # Create refined mask
                refined_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
                
                # Apply morphological operations
                kernel = np.ones((5, 5), np.uint8)
                refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)
                refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)
                
                return refined_mask
            
            return initial_mask
            
        except Exception as e:
            logger.error(f"Error refining person mask: {str(e)}")
            return initial_mask
    
    def detect_background_objects(self, image: np.ndarray) -> List[Dict]:
        """
        Detect and classify background objects
        
        Args:
            image: Input image
            
        Returns:
            List of detected objects
        """
        segmentation_results = self.segment_image(image)
        
        if not segmentation_results:
            return []
        
        objects = []
        class_masks = segmentation_results.get('class_masks', {})
        
        for class_name, mask in class_masks.items():
            if class_name != 'person' and class_name != '__background__':
                # Find contours for this class
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    if cv2.contourArea(contour) > 1000:  # Minimum area threshold
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        object_data = {
                            'class': class_name,
                            'bbox': (x, y, x + w, y + h),
                            'area': cv2.contourArea(contour),
                            'mask': mask,
                            'center': (x + w // 2, y + h // 2)
                        }
                        
                        objects.append(object_data)
        
        return objects
    
    def create_trimap(self, segmentation_mask: np.ndarray, erode_size: int = 10, 
                     dilate_size: int = 20) -> np.ndarray:
        """
        Create trimap for alpha matting
        
        Args:
            segmentation_mask: Binary segmentation mask
            erode_size: Erosion kernel size
            dilate_size: Dilation kernel size
            
        Returns:
            Trimap (0=background, 128=unknown, 255=foreground)
        """
        try:
            # Create kernels
            erode_kernel = np.ones((erode_size, erode_size), np.uint8)
            dilate_kernel = np.ones((dilate_size, dilate_size), np.uint8)
            
            # Erode for sure foreground
            sure_fg = cv2.erode(segmentation_mask, erode_kernel, iterations=1)
            
            # Dilate for possible foreground
            possible_fg = cv2.dilate(segmentation_mask, dilate_kernel, iterations=1)
            
            # Create trimap
            trimap = np.zeros_like(segmentation_mask)
            trimap[possible_fg == 255] = 128  # Unknown region
            trimap[sure_fg == 255] = 255      # Sure foreground
            # Background remains 0
            
            return trimap
            
        except Exception as e:
            logger.error(f"Error creating trimap: {str(e)}")
            return segmentation_mask
    
    def segment_clothing_details(self, image: np.ndarray, person_mask: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Detailed clothing segmentation within person mask
        
        Args:
            image: Original image
            person_mask: Person segmentation mask
            
        Returns:
            Dictionary of detailed clothing masks
        """
        clothing_details = {}
        
        try:
            # Extract person region
            person_region = cv2.bitwise_and(image, image, mask=person_mask)
            
            # Convert to different color spaces for better segmentation
            hsv = cv2.cvtColor(person_region, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(person_region, cv2.COLOR_BGR2LAB)
            
            # Use k-means clustering to segment clothing colors
            pixel_values = person_region[person_mask > 0].reshape(-1, 3)
            
            if len(pixel_values) > 0:
                from sklearn.cluster import KMeans
                
                # Cluster into clothing regions
                n_clusters = min(5, len(pixel_values))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(pixel_values)
                
                # Create masks for each cluster
                for i in range(n_clusters):
                    cluster_mask = np.zeros_like(person_mask)
                    cluster_pixels = pixel_values[labels == i]
                    
                    if len(cluster_pixels) > 100:  # Minimum pixels threshold
                        # Find pixels belonging to this cluster
                        cluster_indices = np.where(labels == i)[0]
                        person_indices = np.where(person_mask > 0)
                        
                        # Map back to image coordinates
                        for idx in cluster_indices:
                            if idx < len(person_indices[0]):
                                y, x = person_indices[0][idx], person_indices[1][idx]
                                cluster_mask[y, x] = 255
                        
                        # Clean up mask
                        kernel = np.ones((3, 3), np.uint8)
                        cluster_mask = cv2.morphologyEx(cluster_mask, cv2.MORPH_OPEN, kernel)
                        
                        # Determine clothing type based on region and color
                        clothing_type = self._classify_clothing_region(cluster_mask, kmeans.cluster_centers_[i])
                        clothing_details[f"{clothing_type}_{i}"] = cluster_mask
            
        except Exception as e:
            logger.error(f"Error in detailed clothing segmentation: {str(e)}")
        
        return clothing_details
    
    def _classify_clothing_region(self, mask: np.ndarray, avg_color: np.ndarray) -> str:
        """
        Classify clothing region based on location and color
        
        Args:
            mask: Clothing region mask
            avg_color: Average color of the region
            
        Returns:
            Clothing type string
        """
        try:
            # Find region center
            moments = cv2.moments(mask)
            if moments['m00'] == 0:
                return "clothing"
            
            center_y = int(moments['m01'] / moments['m00'])
            center_x = int(moments['m10'] / moments['m00'])
            
            # Get image dimensions
            height, width = mask.shape
            
            # Classify based on position
            relative_y = center_y / height
            
            if relative_y < 0.3:
                return "hat_accessories"
            elif relative_y < 0.6:
                return "upper_clothing"
            elif relative_y < 0.85:
                return "lower_clothing"
            else:
                return "shoes"
                
        except Exception as e:
            logger.error(f"Error classifying clothing region: {str(e)}")
            return "clothing"
    
    def apply_skin_smoothing(self, image: np.ndarray, person_mask: np.ndarray) -> np.ndarray:
        """
        Apply skin smoothing to visible skin areas
        
        Args:
            image: Original image
            person_mask: Person segmentation mask
            
        Returns:
            Image with smoothed skin
        """
        try:
            result = image.copy()
            
            # Detect skin regions
            skin_mask = self._detect_skin_regions(image, person_mask)
            
            if np.any(skin_mask > 0):
                # Apply bilateral filter for skin smoothing
                smoothed = cv2.bilateralFilter(image, 15, 80, 80)
                
                # Blend smoothed areas with original
                skin_mask_3d = np.repeat(skin_mask[:, :, np.newaxis], 3, axis=2) / 255.0
                result = image.astype(np.float32) * (1 - skin_mask_3d) + smoothed.astype(np.float32) * skin_mask_3d
                result = result.astype(np.uint8)
            
            return result
            
        except Exception as e:
            logger.error(f"Error applying skin smoothing: {str(e)}")
            return image
    
    def _detect_skin_regions(self, image: np.ndarray, person_mask: np.ndarray) -> np.ndarray:
        """Detect skin regions within person mask"""
        try:
            # Convert to HSV for skin detection
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Define skin color range in HSV
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            
            # Create skin mask
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Combine with person mask
            skin_mask = cv2.bitwise_and(skin_mask, person_mask)
            
            # Clean up mask
            kernel = np.ones((5, 5), np.uint8)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
            
            return skin_mask
            
        except Exception as e:
            logger.error(f"Error detecting skin regions: {str(e)}")
            return np.zeros_like(person_mask)
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'model_name': self.model_name,
            'device': str(self.device),
            'model_loaded': self.model_loaded,
            'num_classes': len(self.coco_classes),
            'supports_human_parsing': True
        }
    
    def save_model_state(self, save_path: str):
        """Save model state (not the weights, just configuration)"""
        try:
            state = {
                'model_name': self.model_name,
                'device': str(self.device),
                'coco_classes': self.coco_classes,
                'human_parsing_classes': self.human_parsing_classes
            }
            
            with open(save_path, 'wb') as f:
                import pickle
                pickle.dump(state, f)
            
            logger.info(f"Model state saved to: {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving model state: {str(e)}")
    
    def load_model_state(self, load_path: str):
        """Load model state"""
        try:
            with open(load_path, 'rb') as f:
                import pickle
                state = pickle.load(f)
            
            # Update configuration
            self.model_name = state.get('model_name', self.model_name)
            self.coco_classes = state.get('coco_classes', self.coco_classes)
            self.human_parsing_classes = state.get('human_parsing_classes', self.human_parsing_classes)
            
            logger.info(f"Model state loaded from: {load_path}")
            
        except Exception as e:
            logger.error(f"Error loading model state: {str(e)}")
    
    def batch_segment(self, images: List[np.ndarray], batch_size: int = 4) -> List[Dict]:
        """
        Perform batch segmentation for multiple images
        
        Args:
            images: List of input images
            batch_size: Batch size for processing
            
        Returns:
            List of segmentation results
        """
        results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_results = []
            
            for image in batch:
                result = self.segment_image(image)
                batch_results.append(result)
            
            results.extend(batch_results)
        
        return results