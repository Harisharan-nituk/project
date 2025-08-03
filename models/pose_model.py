import numpy as np
import cv2
import mediapipe as mp
from typing import Dict, List, Optional, Tuple
import logging
import pickle
from pathlib import Path
import requests
import os

logger = logging.getLogger(__name__)

class PoseModel:
    """Pose estimation model wrapper for MediaPipe"""
    
    def __init__(self):
        """Initialize pose model"""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Model configuration
        self.model_config = {
            'static_image_mode': False,
            'model_complexity': 1,
            'smooth_landmarks': True,
            'enable_segmentation': True,
            'min_detection_confidence': 0.5,
            'min_tracking_confidence': 0.5
        }
        
        # Initialize pose detector
        self.pose_detector = None
        self._initialize_model()
        
        # Pose connections for drawing
        self.pose_connections = [
            # Face connections
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
            # Upper body connections
            (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
            # Lower body connections
            (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (27, 29), (27, 31),
            (24, 26), (26, 28), (28, 30), (28, 32), (29, 31), (30, 32)
        ]
        
        # Body part groups
        self.body_part_groups = {
            'head': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'left_arm': [11, 13, 15, 17, 19, 21],
            'right_arm': [12, 14, 16, 18, 20, 22],
            'torso': [11, 12, 23, 24],
            'left_leg': [23, 25, 27, 29, 31],
            'right_leg': [24, 26, 28, 30, 32]
        }
        
        # Joint angle calculations
        self.joint_definitions = {
            'left_shoulder': ([11], [13], [15]),
            'right_shoulder': ([12], [14], [16]),
            'left_elbow': ([11], [13], [15]),
            'right_elbow': ([12], [14], [16]),
            'left_hip': ([23], [25], [27]),
            'right_hip': ([24], [26], [28]),
            'left_knee': ([23], [25], [27]),
            'right_knee': ([24], [26], [28])
        }
    
    def _initialize_model(self):
        """Initialize the MediaPipe pose model"""
        try:
            self.pose_detector = self.mp_pose.Pose(**self.model_config)
            logger.info("Pose model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pose model: {str(e)}")
            raise
    
    def detect_pose(self, image: np.ndarray) -> Dict:
        """
        Detect pose in image
        
        Args:
            image: Input image
            
        Returns:
            Pose detection results
        """
        if self.pose_detector is None:
            raise RuntimeError("Pose model not initialized")
        
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process image
            results = self.pose_detector.process(rgb_image)
            
            # Extract pose data
            pose_data = {
                'landmarks': None,
                'world_landmarks': None,
                'segmentation_mask': None,
                'pose_present': False,
                'confidence_scores': None
            }
            
            if results.pose_landmarks:
                pose_data['pose_present'] = True
                pose_data['landmarks'] = self._extract_landmarks(results.pose_landmarks, image.shape)
                pose_data['confidence_scores'] = self._extract_confidence_scores(results.pose_landmarks)
                
                if results.pose_world_landmarks:
                    pose_data['world_landmarks'] = self._extract_world_landmarks(results.pose_world_landmarks)
                
                if results.segmentation_mask is not None:
                    pose_data['segmentation_mask'] = (results.segmentation_mask > 0.5).astype(np.uint8) * 255
            
            return pose_data
            
        except Exception as e:
            logger.error(f"Error in pose detection: {str(e)}")
            return {'pose_present': False}
    
    def _extract_landmarks(self, landmarks, image_shape: Tuple) -> np.ndarray:
        """Extract normalized landmarks and convert to pixel coordinates"""
        height, width = image_shape[:2]
        
        landmark_points = []
        for landmark in landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            landmark_points.append([x, y])
        
        return np.array(landmark_points)
    
    def _extract_confidence_scores(self, landmarks) -> np.ndarray:
        """Extract confidence scores for landmarks"""
        confidence_scores = []
        for landmark in landmarks.landmark:
            confidence_scores.append(landmark.visibility)
        
        return np.array(confidence_scores)
    
    def _extract_world_landmarks(self, world_landmarks) -> np.ndarray:
        """Extract 3D world landmarks"""
        world_points = []
        for landmark in world_landmarks.landmark:
            world_points.append([landmark.x, landmark.y, landmark.z])
        
        return np.array(world_points)
    
    def calculate_joint_angles(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Calculate joint angles from landmarks
        
        Args:
            landmarks: Pose landmarks
            
        Returns:
            Dictionary of joint angles in degrees
        """
        angles = {}
        
        for joint_name, (p1_idx, p2_idx, p3_idx) in self.joint_definitions.items():
            if len(landmarks) > max(max(p1_idx), max(p2_idx), max(p3_idx)):
                p1 = landmarks[p1_idx[0]]
                p2 = landmarks[p2_idx[0]]
                p3 = landmarks[p3_idx[0]]
                
                angle = self._calculate_angle(p1, p2, p3)
                angles[joint_name] = angle
        
        return angles
    
    def _calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle between three points"""
        # Vectors
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Calculate angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        angle = np.arccos(cos_angle)
        return np.degrees(angle)
    
    def get_pose_classification(self, landmarks: np.ndarray) -> str:
        """
        Classify pose type based on landmarks
        
        Args:
            landmarks: Pose landmarks
            
        Returns:
            Pose classification string
        """
        if len(landmarks) < 33:
            return "unknown"
        
        # Calculate key angles
        angles = self.calculate_joint_angles(landmarks)
        
        # Simple pose classification
        if angles.get('left_elbow', 180) < 90 and angles.get('right_elbow', 180) < 90:
            return "arms_raised"
        elif landmarks[15][1] < landmarks[11][1] or landmarks[16][1] < landmarks[12][1]:
            return "hands_up"
        elif abs(landmarks[11][0] - landmarks[12][0]) > abs(landmarks[23][0] - landmarks[24][0]) * 1.5:
            return "arms_spread"
        elif landmarks[27][1] < landmarks[25][1] or landmarks[28][1] < landmarks[26][1]:
            return "leg_raised"
        else:
            return "standing"
    
    def save_model_weights(self, weights_path: str):
        """Save learned model weights"""
        # MediaPipe models are pre-trained, so we save configuration instead
        config_data = {
            'model_config': self.model_config,
            'body_part_groups': self.body_part_groups,
            'joint_definitions': self.joint_definitions
        }
        
        with open(weights_path, 'wb') as f:
            pickle.dump(config_data, f)
        
        logger.info(f"Model configuration saved to: {weights_path}")
    
    def load_model_weights(self, weights_path: str):
        """Load model weights/configuration"""
        try:
            with open(weights_path, 'rb') as f:
                config_data = pickle.load(f)
            
            self.model_config.update(config_data.get('model_config', {}))
            self.body_part_groups.update(config_data.get('body_part_groups', {}))
            self.joint_definitions.update(config_data.get('joint_definitions', {}))
            
            # Reinitialize with new config
            self._initialize_model()
            
            logger.info(f"Model configuration loaded from: {weights_path}")
            
        except Exception as e:
            logger.error(f"Error loading model weights: {str(e)}")
    
    def download_model_files(self, model_url: str, save_path: str):
        """Download model files if needed"""
        try:
            if not os.path.exists(save_path):
                logger.info(f"Downloading model from: {model_url}")
                
                response = requests.get(model_url, stream=True)
                response.raise_for_status()
                
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                logger.info(f"Model downloaded to: {save_path}")
            else:
                logger.info(f"Model already exists: {save_path}")
                
        except Exception as e:
            logger.error(f"Error downloading model: {str(e)}")
            raise