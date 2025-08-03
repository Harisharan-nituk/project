import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import os
import json
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

class ClothingChanger:
    """Advanced clothing manipulation and replacement system"""
    
    def __init__(self, segmentation_model=None):
        """Initialize clothing changer"""
        self.segmentation_model = segmentation_model
        
        # Clothing categories and their corresponding body parts
        self.clothing_mapping = {
            'shirt': ['torso', 'left_arm', 'right_arm'],
            'pants': ['left_leg', 'right_leg', 'hips'],
            'dress': ['torso', 'left_arm', 'right_arm', 'left_leg', 'right_leg'],
            'jacket': ['torso', 'left_arm', 'right_arm'],
            'skirt': ['hips', 'left_leg', 'right_leg'],
            'shorts': ['hips', 'left_leg', 'right_leg'],
            'tank_top': ['torso'],
            'long_sleeve': ['torso', 'left_arm', 'right_arm'],
            'accessories': ['head', 'torso']
        }
        
        # Body part segmentation indices (assuming standard human parsing)
        self.body_parts_indices = {
            'head': [1, 2, 4, 13],  # Head, hair, hat, face
            'torso': [3, 6],        # Upper clothes, dress
            'left_arm': [14],       # Left arm
            'right_arm': [15],      # Right arm
            'left_leg': [16, 18],   # Left leg, left shoe
            'right_leg': [17, 19],  # Right leg, right shoe
            'hips': [9, 12]         # Lower clothes, pants
        }
        
        # Clothing templates cache
        self.clothing_templates = {}
        self.load_clothing_templates()
        
        # Color palette for clothing
        self.color_palettes = {
            'formal': [(0, 0, 0), (255, 255, 255), (128, 128, 128), (0, 0, 128)],
            'casual': [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)],
            'summer': [(255, 192, 203), (255, 165, 0), (255, 255, 224), (173, 216, 230)],
            'winter': [(128, 0, 0), (0, 0, 128), (128, 128, 128), (0, 0, 0)]
        }
    
    def change_clothing(self, frame: np.ndarray, clothing_style: str, pose_data: Dict) -> np.ndarray:
        """
        Change clothing in the frame based on style and pose
        
        Args:
            frame: Input frame
            clothing_style: Style of clothing to apply
            pose_data: Pose data for proper fitting
            
        Returns:
            Frame with changed clothing
        """
        try:
            logger.info(f"Changing clothing to style: {clothing_style}")
            
            # Segment the person and body parts
            segmentation_mask = self._segment_person(frame)
            body_parts_mask = self._segment_body_parts(frame, segmentation_mask)
            
            # Get clothing template
            clothing_template = self._get_clothing_template(clothing_style)
            if clothing_template is None:
                logger.warning(f"No template found for clothing style: {clothing_style}")
                return frame
            
            # Apply clothing based on pose
            result_frame = self._apply_clothing_with_pose(
                frame, 
                clothing_template, 
                body_parts_mask, 
                pose_data
            )
            
            # Post-process for realism
            result_frame = self._post_process_clothing(result_frame, frame, segmentation_mask)
            
            return result_frame
            
        except Exception as e:
            logger.error(f"Error changing clothing: {str(e)}")
            return frame
    
    def _segment_person(self, frame: np.ndarray) -> np.ndarray:
        """
        Segment person from background
        
        Args:
            frame: Input frame
            
        Returns:
            Person segmentation mask
        """
        if self.segmentation_model is None:
            # Fallback: simple background subtraction or use pose data
            return self._simple_person_segmentation(frame)
        
        # Use advanced segmentation model
        return self.segmentation_model.segment_person(frame)
    
    def _simple_person_segmentation(self, frame: np.ndarray) -> np.ndarray:
        """
        Simple person segmentation using color and edge detection
        
        Args:
            frame: Input frame
            
        Returns:
            Simple person mask
        """
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define skin color range (approximate)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create skin mask
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find largest contour (assuming it's the person)
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            person_mask = np.zeros_like(skin_mask)
            cv2.fillPoly(person_mask, [largest_contour], 255)
            
            # Expand mask to include clothing
            kernel = np.ones((20, 20), np.uint8)
            person_mask = cv2.dilate(person_mask, kernel, iterations=1)
            
            return person_mask
        
        return np.zeros(frame.shape[:2], dtype=np.uint8)
    
    def _segment_body_parts(self, frame: np.ndarray, person_mask: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Segment different body parts
        
        Args:
            frame: Input frame
            person_mask: Person segmentation mask
            
        Returns:
            Dictionary of body part masks
        """
        body_parts = {}
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Create approximate body part masks based on typical human proportions
        # Head region (top 1/8 of person)
        head_mask = np.zeros_like(person_mask)
        head_region = person_mask.copy()
        head_region[int(height * 0.125):] = 0
        body_parts['head'] = head_region
        
        # Torso region (1/8 to 5/8 of person height)
        torso_mask = person_mask.copy()
        torso_mask[:int(height * 0.125)] = 0
        torso_mask[int(height * 0.625):] = 0
        body_parts['torso'] = torso_mask
        
        # Arms (side regions of torso)
        arm_width = int(width * 0.15)
        
        # Left arm
        left_arm_mask = torso_mask.copy()
        left_arm_mask[:, arm_width:] = 0
        body_parts['left_arm'] = left_arm_mask
        
        # Right arm
        right_arm_mask = torso_mask.copy()
        right_arm_mask[:, :-arm_width] = 0
        body_parts['right_arm'] = right_arm_mask
        
        # Hips region (5/8 to 3/4 of person height)
        hips_mask = person_mask.copy()
        hips_mask[:int(height * 0.625)] = 0
        hips_mask[int(height * 0.75):] = 0
        body_parts['hips'] = hips_mask
        
        # Legs (3/4 to bottom of person)
        legs_mask = person_mask.copy()
        legs_mask[:int(height * 0.75)] = 0
        
        # Split legs
        center_x = width // 2
        
        left_leg_mask = legs_mask.copy()
        left_leg_mask[:, center_x:] = 0
        body_parts['left_leg'] = left_leg_mask
        
        right_leg_mask = legs_mask.copy()
        right_leg_mask[:, :center_x] = 0
        body_parts['right_leg'] = right_leg_mask
        
        return body_parts
    
    def _get_clothing_template(self, clothing_style: str) -> Optional[Dict]:
        """
        Get clothing template for the specified style
        
        Args:
            clothing_style: Style identifier
            
        Returns:
            Clothing template dictionary or None
        """
        if clothing_style in self.clothing_templates:
            return self.clothing_templates[clothing_style]
        
        # Generate default template
        return self._generate_default_template(clothing_style)
    
    def _generate_default_template(self, clothing_style: str) -> Dict:
        """
        Generate default clothing template
        
        Args:
            clothing_style: Style identifier
            
        Returns:
            Default clothing template
        """
        # Determine clothing type from style name
        clothing_type = 'shirt'  # default
        
        if any(keyword in clothing_style.lower() for keyword in ['dress', 'gown']):
            clothing_type = 'dress'
        elif any(keyword in clothing_style.lower() for keyword in ['pants', 'jeans', 'trousers']):
            clothing_type = 'pants'
        elif any(keyword in clothing_style.lower() for keyword in ['jacket', 'blazer', 'coat']):
            clothing_type = 'jacket'
        elif any(keyword in clothing_style.lower() for keyword in ['skirt']):
            clothing_type = 'skirt'
        elif any(keyword in clothing_style.lower() for keyword in ['shorts']):
            clothing_type = 'shorts'
        
        # Get color palette
        palette = 'casual'
        if 'formal' in clothing_style.lower():
            palette = 'formal'
        elif 'summer' in clothing_style.lower():
            palette = 'summer'
        elif 'winter' in clothing_style.lower():
            palette = 'winter'
        
        template = {
            'type': clothing_type,
            'body_parts': self.clothing_mapping.get(clothing_type, ['torso']),
            'colors': self.color_palettes[palette],
            'texture': 'solid',
            'pattern': None,
            'style_attributes': {
                'fit': 'regular',
                'length': 'regular',
                'sleeves': 'regular' if clothing_type in ['shirt', 'dress', 'jacket'] else None
            }
        }
        
        return template
    
    def _apply_clothing_with_pose(self, frame: np.ndarray, clothing_template: Dict,
                                 body_parts_mask: Dict[str, np.ndarray], pose_data: Dict) -> np.ndarray:
        """
        Apply clothing template considering pose data
        
        Args:
            frame: Input frame
            clothing_template: Clothing template
            body_parts_mask: Body parts segmentation
            pose_data: Pose information
            
        Returns:
            Frame with applied clothing
        """
        result_frame = frame.copy()
        
        # Get body parts to modify
        target_parts = clothing_template['body_parts']
        clothing_colors = clothing_template['colors']
        
        for part_name in target_parts:
            if part_name in body_parts_mask:
                part_mask = body_parts_mask[part_name]
                
                # Apply clothing to this body part
                result_frame = self._apply_clothing_to_part(
                    result_frame,
                    part_mask,
                    clothing_colors[0],  # Use primary color
                    clothing_template,
                    pose_data
                )
        
        return result_frame
    
    def _apply_clothing_to_part(self, frame: np.ndarray, part_mask: np.ndarray, 
                               color: Tuple[int, int, int], template: Dict, pose_data: Dict) -> np.ndarray:
        """
        Apply clothing to a specific body part
        
        Args:
            frame: Input frame
            part_mask: Mask for the body part
            color: Clothing color
            template: Clothing template
            pose_data: Pose data
            
        Returns:
            Frame with clothing applied to the part
        """
        # Create clothing overlay
        clothing_overlay = np.zeros_like(frame)
        
        # Fill the masked region with clothing color
        clothing_overlay[part_mask > 0] = color
        
        # Add texture if specified
        if template.get('texture') == 'fabric':
            clothing_overlay = self._add_fabric_texture(clothing_overlay, part_mask)
        elif template.get('texture') == 'denim':
            clothing_overlay = self._add_denim_texture(clothing_overlay, part_mask)
        
        # Add pattern if specified
        if template.get('pattern'):
            clothing_overlay = self._add_pattern(clothing_overlay, part_mask, template['pattern'])
        
        # Apply shading based on pose
        clothing_overlay = self._apply_pose_shading(clothing_overlay, part_mask, pose_data)
        
        # Blend with original frame
        alpha = 0.8  # Clothing opacity
        mask_3d = np.repeat(part_mask[:, :, np.newaxis], 3, axis=2) / 255.0
        
        result = frame * (1 - mask_3d * alpha) + clothing_overlay * mask_3d * alpha
        
        return result.astype(np.uint8)
    
    def _add_fabric_texture(self, clothing_overlay: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Add fabric-like texture to clothing"""
        # Create noise pattern
        noise = np.random.randint(-20, 20, clothing_overlay.shape[:2])
        
        # Apply noise only to masked regions
        for c in range(3):
            channel = clothing_overlay[:, :, c].astype(np.int16)
            channel[mask > 0] = np.clip(channel[mask > 0] + noise[mask > 0], 0, 255)
            clothing_overlay[:, :, c] = channel.astype(np.uint8)
        
        return clothing_overlay
    
    def _add_denim_texture(self, clothing_overlay: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Add denim-like texture to clothing"""
        # Create horizontal lines pattern
        height = clothing_overlay.shape[0]
        for y in range(0, height, 3):
            if y < height:
                line_mask = mask[y, :] > 0
                clothing_overlay[y, line_mask] = np.clip(
                    clothing_overlay[y, line_mask].astype(np.int16) + 15, 0, 255
                ).astype(np.uint8)
        
        return clothing_overlay
    
    def _add_pattern(self, clothing_overlay: np.ndarray, mask: np.ndarray, pattern: str) -> np.ndarray:
        """Add pattern to clothing"""
        if pattern == 'stripes':
            return self._add_stripes_pattern(clothing_overlay, mask)
        elif pattern == 'dots':
            return self._add_dots_pattern(clothing_overlay, mask)
        elif pattern == 'plaid':
            return self._add_plaid_pattern(clothing_overlay, mask)
        
        return clothing_overlay
    
    def _add_stripes_pattern(self, clothing_overlay: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Add stripes pattern"""
        height = clothing_overlay.shape[0]
        stripe_width = 10
        
        for y in range(0, height, stripe_width * 2):
            for stripe_y in range(y, min(y + stripe_width, height)):
                if stripe_y < height:
                    stripe_mask = mask[stripe_y, :] > 0
                    clothing_overlay[stripe_y, stripe_mask] = np.clip(
                        clothing_overlay[stripe_y, stripe_mask].astype(np.int16) + 30, 0, 255
                    ).astype(np.uint8)
        
        return clothing_overlay
    
    def _add_dots_pattern(self, clothing_overlay: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Add polka dots pattern"""
        height, width = clothing_overlay.shape[:2]
        dot_spacing = 20
        dot_radius = 3
        
        for y in range(dot_spacing, height, dot_spacing):
            for x in range(dot_spacing, width, dot_spacing):
                if mask[y, x] > 0:
                    cv2.circle(clothing_overlay, (x, y), dot_radius, (255, 255, 255), -1)
        
        return clothing_overlay
    
    def _add_plaid_pattern(self, clothing_overlay: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Add plaid pattern"""
        # Combine vertical and horizontal stripes
        clothing_overlay = self._add_stripes_pattern(clothing_overlay, mask)
        
        # Add vertical stripes
        width = clothing_overlay.shape[1]
        stripe_width = 10
        
        for x in range(0, width, stripe_width * 2):
            for stripe_x in range(x, min(x + stripe_width, width)):
                if stripe_x < width:
                    stripe_mask = mask[:, stripe_x] > 0
                    clothing_overlay[stripe_mask, stripe_x] = np.clip(
                        clothing_overlay[stripe_mask, stripe_x].astype(np.int16) + 20, 0, 255
                    ).astype(np.uint8)
        
        return clothing_overlay
    
    def _apply_pose_shading(self, clothing_overlay: np.ndarray, mask: np.ndarray, pose_data: Dict) -> np.ndarray:
        """Apply shading based on pose to make clothing look more realistic"""
        if not pose_data.get('pose_present') or pose_data.get('landmarks') is None:
            return clothing_overlay
        
        # Get pose angles for shading
        angles = self._calculate_shading_angles(pose_data)
        
        # Apply gradient shading based on body orientation
        height, width = clothing_overlay.shape[:2]
        
        # Create gradient based on lighting direction
        y_gradient = np.linspace(1.0, 0.7, height)
        x_gradient = np.linspace(0.9, 1.1, width)
        
        # Combine gradients
        gradient = np.outer(y_gradient, x_gradient)
        gradient = np.repeat(gradient[:, :, np.newaxis], 3, axis=2)
        
        # Apply gradient only to masked regions
        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2) / 255.0
        clothing_overlay = clothing_overlay.astype(np.float32)
        clothing_overlay = clothing_overlay * (1 - mask_3d) + clothing_overlay * gradient * mask_3d
        
        return clothing_overlay.astype(np.uint8)
    
    def _calculate_shading_angles(self, pose_data: Dict) -> Dict[str, float]:
        """Calculate angles for realistic shading"""
        landmarks = pose_data['landmarks']
        angles = {}
        
        # Calculate torso angle
        if len(landmarks) > 24:
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            
            # Calculate torso tilt
            shoulder_vector = right_shoulder - left_shoulder
            hip_vector = right_hip - left_hip
            
            angles['torso_tilt'] = np.arctan2(shoulder_vector[1], shoulder_vector[0])
            angles['hip_tilt'] = np.arctan2(hip_vector[1], hip_vector[0])
        
        return angles
    
    def _post_process_clothing(self, result_frame: np.ndarray, original_frame: np.ndarray, 
                              person_mask: np.ndarray) -> np.ndarray:
        """Post-process clothing for realism"""
        # Apply slight blur to clothing edges for more natural look
        kernel = np.ones((3, 3), np.uint8)
        smooth_mask = cv2.morphologyEx(person_mask, cv2.MORPH_OPEN, kernel)
        
        # Create edge mask
        edge_mask = cv2.Canny(smooth_mask, 50, 150)
        edge_mask = cv2.dilate(edge_mask, kernel, iterations=1)
        
        # Apply Gaussian blur to edges
        blurred_result = cv2.GaussianBlur(result_frame, (3, 3), 0)
        
        # Blend edges
        edge_mask_3d = np.repeat(edge_mask[:, :, np.newaxis], 3, axis=2) / 255.0
        final_result = result_frame * (1 - edge_mask_3d) + blurred_result * edge_mask_3d
        
        return final_result.astype(np.uint8)
    
    def load_clothing_templates(self):
        """Load clothing templates from file"""
        templates_file = Path("data/clothing/templates.json")
        
        if templates_file.exists():
            try:
                with open(templates_file, 'r') as f:
                    self.clothing_templates = json.load(f)
                logger.info(f"Loaded {len(self.clothing_templates)} clothing templates")
            except Exception as e:
                logger.error(f"Error loading clothing templates: {str(e)}")
        else:
            # Create default templates
            self._create_default_templates()
    
    def _create_default_templates(self):
        """Create default clothing templates"""
        self.clothing_templates = {
            'formal_suit': {
                'type': 'jacket',
                'body_parts': ['torso', 'left_arm', 'right_arm'],
                'colors': [(0, 0, 0), (255, 255, 255)],
                'texture': 'fabric',
                'pattern': None,
                'style_attributes': {'fit': 'fitted', 'length': 'regular', 'sleeves': 'long'}
            },
            'casual_wear': {
                'type': 'shirt',
                'body_parts': ['torso'],
                'colors': [(0, 0, 255), (255, 255, 255)],
                'texture': 'fabric',
                'pattern': None,
                'style_attributes': {'fit': 'regular', 'length': 'regular', 'sleeves': 'short'}
            },
            'dress': {
                'type': 'dress',
                'body_parts': ['torso', 'hips', 'left_leg', 'right_leg'],
                'colors': [(255, 192, 203), (255, 255, 255)],
                'texture': 'fabric',
                'pattern': 'dots',
                'style_attributes': {'fit': 'fitted', 'length': 'knee', 'sleeves': 'short'}
            }
        }
    
    def save_clothing_templates(self):
        """Save clothing templates to file"""
        templates_file = Path("data/clothing/templates.json")
        templates_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(templates_file, 'w') as f:
                json.dump(self.clothing_templates, f, indent=2)
            logger.info("Clothing templates saved successfully")
        except Exception as e:
            logger.error(f"Error saving clothing templates: {str(e)}")
    
    def add_custom_clothing_template(self, name: str, template: Dict):
        """Add custom clothing template"""
        self.clothing_templates[name] = template
        self.save_clothing_templates()
        logger.info(f"Added custom clothing template: {name}")
    
    def get_available_styles(self) -> List[str]:
        """Get list of available clothing styles"""
        return list(self.clothing_templates.keys())