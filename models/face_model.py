import cv2
import numpy as np
import face_recognition
import logging
from typing import Dict, List, Optional, Tuple
import pickle
from pathlib import Path
import os
import requests

logger = logging.getLogger(__name__)

class FaceModel:
    """Face detection and recognition model"""
    
    def __init__(self):
        """Initialize face model"""
        self.face_detection_model = "hog"  # or "cnn" for better accuracy
        self.face_recognition_tolerance = 0.6
        self.num_jitters = 1
        self.model_loaded = False
        
        # Face cascade for backup detection
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            logger.info("OpenCV face cascade loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load face cascade: {str(e)}")
            self.face_cascade = None
        
        # Initialize dlib components if available
        self.dlib_available = False
        self.face_detector = None
        self.shape_predictor = None
        self._initialize_dlib()
        
        # Known faces database
        self.known_faces = {}
        self.known_face_encodings = []
        self.known_face_names = []
        
        self.model_loaded = True
        logger.info("Face model initialized successfully")
    
    def _initialize_dlib(self):
        """Initialize dlib components"""
        try:
            import dlib
            
            # Initialize face detector
            self.face_detector = dlib.get_frontal_face_detector()
            
            # Check for shape predictor
            predictor_path = "models/weights/shape_predictor_68_face_landmarks.dat"
            if os.path.exists(predictor_path):
                self.shape_predictor = dlib.shape_predictor(predictor_path)
                self.dlib_available = True
                logger.info("Dlib components initialized successfully")
            else:
                logger.warning(f"Shape predictor not found at: {predictor_path}")
                logger.info("Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
                
        except ImportError:
            logger.warning("Dlib not available, using basic face detection")
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in image
        
        Args:
            image: Input image
            
        Returns:
            List of detected face data
        """
        faces = []
        
        try:
            # Primary detection using face_recognition
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(
                rgb_image, 
                model=self.face_detection_model
            )
            
            # Extract face encodings
            face_encodings = face_recognition.face_encodings(
                rgb_image, 
                face_locations,
                num_jitters=self.num_jitters
            )
            
            # Process each detected face
            for i, (encoding, location) in enumerate(zip(face_encodings, face_locations)):
                top, right, bottom, left = location
                
                face_data = {
                    'id': i,
                    'bbox': (left, top, right, bottom),
                    'center': ((left + right) // 2, (top + bottom) // 2),
                    'size': (right - left, bottom - top),
                    'encoding': encoding,
                    'confidence': 1.0,  # face_recognition doesn't provide confidence
                    'landmarks': None,
                    'quality_score': self._calculate_face_quality(image, (left, top, right, bottom))
                }
                
                # Add landmarks if dlib is available
                if self.dlib_available and self.shape_predictor is not None:
                    landmarks = self._extract_landmarks(image, (left, top, right, bottom))
                    face_data['landmarks'] = landmarks
                
                faces.append(face_data)
            
            # Fallback to OpenCV cascade if no faces found
            if not faces and self.face_cascade is not None:
                faces = self._detect_faces_cascade(image)
            
            return faces
            
        except Exception as e:
            logger.error(f"Error in face detection: {str(e)}")
            return []
    
    def _detect_faces_cascade(self, image: np.ndarray) -> List[Dict]:
        """Fallback face detection using OpenCV cascade"""
        faces = []
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detected_faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            for i, (x, y, w, h) in enumerate(detected_faces):
                face_data = {
                    'id': i,
                    'bbox': (x, y, x + w, y + h),
                    'center': (x + w // 2, y + h // 2),
                    'size': (w, h),
                    'encoding': None,
                    'confidence': 0.8,  # Default confidence for cascade
                    'landmarks': None,
                    'quality_score': self._calculate_face_quality(image, (x, y, x + w, y + h))
                }
                
                faces.append(face_data)
            
            return faces
            
        except Exception as e:
            logger.error(f"Error in cascade face detection: {str(e)}")
            return []
    
    def _extract_landmarks(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Extract facial landmarks using dlib"""
        try:
            if not self.dlib_available or self.shape_predictor is None:
                return None
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            left, top, right, bottom = bbox
            
            # Create dlib rectangle
            import dlib
            rect = dlib.rectangle(left, top, right, bottom)
            
            # Get landmarks
            landmarks = self.shape_predictor(gray, rect)
            
            # Convert to numpy array
            points = []
            for i in range(68):
                point = landmarks.part(i)
                points.append([point.x, point.y])
            
            return np.array(points)
            
        except Exception as e:
            logger.error(f"Error extracting landmarks: {str(e)}")
            return None
    
    def _calculate_face_quality(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """Calculate face quality score"""
        try:
            left, top, right, bottom = bbox
            face_region = image[top:bottom, left:right]
            
            if face_region.size == 0:
                return 0.0
            
            # Convert to grayscale
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Calculate sharpness using Laplacian variance
            laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            
            # Calculate brightness
            brightness = np.mean(gray_face)
            
            # Calculate contrast
            contrast = gray_face.std()
            
            # Normalize scores
            sharpness_score = min(laplacian_var / 500.0, 1.0)
            brightness_score = 1.0 - abs(brightness - 127.5) / 127.5
            contrast_score = min(contrast / 127.5, 1.0)
            
            # Combine scores
            quality_score = (sharpness_score + brightness_score + contrast_score) / 3.0
            
            return quality_score
            
        except Exception as e:
            logger.error(f"Error calculating face quality: {str(e)}")
            return 0.5
    
    def recognize_face(self, face_encoding: np.ndarray, tolerance: float = None) -> Optional[str]:
        """
        Recognize face using known encodings
        
        Args:
            face_encoding: Face encoding to match
            tolerance: Recognition tolerance
            
        Returns:
            Name of recognized person or None
        """
        if tolerance is None:
            tolerance = self.face_recognition_tolerance
        
        if not self.known_face_encodings:
            return None
        
        # Compare with known faces
        matches = face_recognition.compare_faces(
            self.known_face_encodings, 
            face_encoding, 
            tolerance=tolerance
        )
        
        # Find best match
        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
        
        if matches and len(matches) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                return self.known_face_names[best_match_index]
        
        return None
    
    def add_known_face(self, image_path: str, person_name: str):
        """
        Add a known face to the database
        
        Args:
            image_path: Path to face image
            person_name: Name of the person
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return False
            
            # Detect faces
            faces = self.detect_faces(image)
            
            if not faces:
                logger.error(f"No faces detected in: {image_path}")
                return False
            
            # Use the best quality face
            best_face = max(faces, key=lambda f: f['quality_score'])
            
            if best_face['encoding'] is not None:
                self.known_face_encodings.append(best_face['encoding'])
                self.known_face_names.append(person_name)
                self.known_faces[person_name] = {
                    'encoding': best_face['encoding'],
                    'image_path': image_path,
                    'quality_score': best_face['quality_score']
                }
                
                logger.info(f"Added known face: {person_name}")
                return True
            else:
                logger.error(f"Could not extract face encoding from: {image_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error adding known face: {str(e)}")
            return False
    
    def save_known_faces(self, save_path: str):
        """Save known faces database"""
        try:
            data = {
                'known_faces': self.known_faces,
                'known_face_encodings': self.known_face_encodings,
                'known_face_names': self.known_face_names
            }
            
            with open(save_path, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"Known faces saved to: {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving known faces: {str(e)}")
    
    def load_known_faces(self, load_path: str):
        """Load known faces database"""
        try:
            with open(load_path, 'rb') as f:
                data = pickle.load(f)
            
            self.known_faces = data.get('known_faces', {})
            self.known_face_encodings = data.get('known_face_encodings', [])
            self.known_face_names = data.get('known_face_names', [])
            
            logger.info(f"Loaded {len(self.known_faces)} known faces from: {load_path}")
            
        except Exception as e:
            logger.error(f"Error loading known faces: {str(e)}")
    
    def align_face(self, image: np.ndarray, landmarks: np.ndarray, 
                   output_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
        """
        Align face using landmarks
        
        Args:
            image: Input image
            landmarks: Facial landmarks
            output_size: Desired output size
            
        Returns:
            Aligned face image
        """
        try:
            if landmarks is None or len(landmarks) < 68:
                return cv2.resize(image, output_size)
            
            # Get eye coordinates
            left_eye = landmarks[36:42].mean(axis=0)
            right_eye = landmarks[42:48].mean(axis=0)
            
            # Calculate angle between eyes
            eye_vector = right_eye - left_eye
            angle = np.degrees(np.arctan2(eye_vector[1], eye_vector[0]))
            
            # Calculate center between eyes
            eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
            
            # Get rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
            
            # Apply rotation
            rotated = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
            
            # Crop and resize to output size
            # Recalculate eye positions after rotation
            ones = np.ones(shape=(len(landmarks), 1))
            points_ones = np.hstack([landmarks, ones])
            transformed_landmarks = rotation_matrix.dot(points_ones.T).T
            
            # Get new eye positions
            new_left_eye = transformed_landmarks[36:42].mean(axis=0)
            new_right_eye = transformed_landmarks[42:48].mean(axis=0)
            new_eye_center = ((new_left_eye[0] + new_right_eye[0]) // 2, 
                             (new_left_eye[1] + new_right_eye[1]) // 2)
            
            # Calculate crop region
            eye_distance = np.linalg.norm(new_right_eye - new_left_eye)
            crop_size = int(eye_distance * 3)  # 3x eye distance for full face
            
            x1 = max(0, int(new_eye_center[0] - crop_size // 2))
            y1 = max(0, int(new_eye_center[1] - crop_size // 2))
            x2 = min(rotated.shape[1], x1 + crop_size)
            y2 = min(rotated.shape[0], y1 + crop_size)
            
            # Crop face
            cropped_face = rotated[y1:y2, x1:x2]
            
            # Resize to output size
            aligned_face = cv2.resize(cropped_face, output_size)
            
            return aligned_face
            
        except Exception as e:
            logger.error(f"Error in face alignment: {str(e)}")
            return cv2.resize(image, output_size)
    
    def extract_face_features(self, face_image: np.ndarray) -> Dict:
        """
        Extract various face features
        
        Args:
            face_image: Face image
            
        Returns:
            Dictionary of face features
        """
        features = {}
        
        try:
            # Detect faces in the image
            faces = self.detect_faces(face_image)
            
            if not faces:
                return features
            
            best_face = faces[0]  # Use first/best face
            
            # Basic features
            features['bbox'] = best_face['bbox']
            features['size'] = best_face['size']
            features['quality_score'] = best_face['quality_score']
            
            # Extract face region
            left, top, right, bottom = best_face['bbox']
            face_region = face_image[top:bottom, left:right]
            
            # Color analysis
            features['average_color'] = np.mean(face_region, axis=(0, 1))
            features['skin_tone'] = self._analyze_skin_tone(face_region)
            
            # Geometric features
            if best_face['landmarks'] is not None:
                landmarks = best_face['landmarks']
                features['landmarks'] = landmarks
                features['face_shape'] = self._analyze_face_shape(landmarks)
                features['eye_aspect_ratio'] = self._calculate_eye_aspect_ratio(landmarks)
                features['mouth_aspect_ratio'] = self._calculate_mouth_aspect_ratio(landmarks)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting face features: {str(e)}")
            return {}
    
    def _analyze_skin_tone(self, face_region: np.ndarray) -> str:
        """Analyze skin tone category"""
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
            
            # Calculate average hue and saturation
            avg_hue = np.mean(hsv[:, :, 0])
            avg_saturation = np.mean(hsv[:, :, 1])
            avg_value = np.mean(hsv[:, :, 2])
            
            # Simple skin tone classification
            if avg_value < 100:
                return "dark"
            elif avg_value < 180:
                return "medium"
            else:
                return "light"
                
        except Exception as e:
            logger.error(f"Error analyzing skin tone: {str(e)}")
            return "unknown"
    
    def _analyze_face_shape(self, landmarks: np.ndarray) -> str:
        """Analyze face shape from landmarks"""
        try:
            # Calculate face dimensions
            # Jaw width (landmarks 0-16)
            jaw_width = np.linalg.norm(landmarks[16] - landmarks[0])
            
            # Face height (top to bottom)
            face_height = np.linalg.norm(landmarks[8] - landmarks[27])
            
            # Cheekbone width (landmarks 1-15)
            cheekbone_width = np.linalg.norm(landmarks[15] - landmarks[1])
            
            # Forehead width (estimated)
            forehead_width = np.linalg.norm(landmarks[26] - landmarks[17])
            
            # Calculate ratios
            width_height_ratio = jaw_width / face_height
            jaw_cheekbone_ratio = jaw_width / cheekbone_width
            
            # Classify face shape
            if width_height_ratio > 1.2:
                return "round"
            elif width_height_ratio < 0.8:
                return "long"
            elif jaw_cheekbone_ratio > 0.9:
                return "square"
            elif jaw_cheekbone_ratio < 0.7:
                return "heart"
            else:
                return "oval"
                
        except Exception as e:
            logger.error(f"Error analyzing face shape: {str(e)}")
            return "unknown"
    
    def _calculate_eye_aspect_ratio(self, landmarks: np.ndarray) -> float:
        """Calculate eye aspect ratio (for blink detection)"""
        try:
            # Left eye landmarks (36-41)
            left_eye = landmarks[36:42]
            
            # Calculate distances
            # Vertical distances
            A = np.linalg.norm(left_eye[1] - left_eye[5])
            B = np.linalg.norm(left_eye[2] - left_eye[4])
            
            # Horizontal distance
            C = np.linalg.norm(left_eye[0] - left_eye[3])
            
            # Eye aspect ratio
            ear = (A + B) / (2.0 * C)
            
            return ear
            
        except Exception as e:
            logger.error(f"Error calculating eye aspect ratio: {str(e)}")
            return 0.3  # Default EAR
    
    def _calculate_mouth_aspect_ratio(self, landmarks: np.ndarray) -> float:
        """Calculate mouth aspect ratio"""
        try:
            # Mouth landmarks (48-67)
            mouth = landmarks[48:68]
            
            # Calculate distances
            # Vertical distances
            A = np.linalg.norm(mouth[2] - mouth[10])  # 50-58
            B = np.linalg.norm(mouth[4] - mouth[8])   # 52-56
            
            # Horizontal distance
            C = np.linalg.norm(mouth[0] - mouth[6])   # 48-54
            
            # Mouth aspect ratio
            mar = (A + B) / (2.0 * C)
            
            return mar
            
        except Exception as e:
            logger.error(f"Error calculating mouth aspect ratio: {str(e)}")
            return 0.5  # Default MAR
    
    def enhance_face_image(self, face_image: np.ndarray) -> np.ndarray:
        """
        Enhance face image quality
        
        Args:
            face_image: Input face image
            
        Returns:
            Enhanced face image
        """
        try:
            # Apply bilateral filter for noise reduction
            enhanced = cv2.bilateralFilter(face_image, 9, 75, 75)
            
            # Enhance contrast using CLAHE
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Apply subtle sharpening
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            
            # Blend original and sharpened
            result = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
            
            return result
            
        except Exception as e:
            logger.error(f"Error enhancing face image: {str(e)}")
            return face_image
    
    def match_lighting_conditions(self, source_face: np.ndarray, target_face: np.ndarray) -> np.ndarray:
        """
        Match lighting conditions between source and target faces
        
        Args:
            source_face: Source face image
            target_face: Target face image for reference
            
        Returns:
            Source face with matched lighting
        """
        try:
            # Convert to LAB color space
            source_lab = cv2.cvtColor(source_face, cv2.COLOR_BGR2LAB)
            target_lab = cv2.cvtColor(target_face, cv2.COLOR_BGR2LAB)
            
            # Calculate statistics for L channel
            source_l_mean = np.mean(source_lab[:, :, 0])
            source_l_std = np.std(source_lab[:, :, 0])
            
            target_l_mean = np.mean(target_lab[:, :, 0])
            target_l_std = np.std(target_lab[:, :, 0])
            
            # Apply histogram matching to L channel
            source_lab[:, :, 0] = ((source_lab[:, :, 0] - source_l_mean) * 
                                  (target_l_std / source_l_std)) + target_l_mean
            
            # Clip values
            source_lab[:, :, 0] = np.clip(source_lab[:, :, 0], 0, 255)
            
            # Convert back to BGR
            matched_face = cv2.cvtColor(source_lab, cv2.COLOR_LAB2BGR)
            
            return matched_face
            
        except Exception as e:
            logger.error(f"Error matching lighting conditions: {str(e)}")
            return source_face
    
    def blend_face_boundary(self, target_image: np.ndarray, source_face: np.ndarray,
                           face_bbox: Tuple[int, int, int, int], blend_ratio: float = 0.1) -> np.ndarray:
        """
        Blend face boundary for seamless integration
        
        Args:
            target_image: Target image
            source_face: Source face to blend
            face_bbox: Face bounding box
            blend_ratio: Ratio of face region to use for blending
            
        Returns:
            Image with blended face boundary
        """
        try:
            left, top, right, bottom = face_bbox
            face_width = right - left
            face_height = bottom - top
            
            # Calculate blend region size
            blend_width = int(face_width * blend_ratio)
            blend_height = int(face_height * blend_ratio)
            
            # Resize source face to match target region
            resized_source = cv2.resize(source_face, (face_width, face_height))
            
            # Create blending mask
            mask = np.ones((face_height, face_width), dtype=np.float32)
            
            # Create gradient at edges
            # Top edge
            for i in range(blend_height):
                alpha = i / blend_height
                mask[i, :] *= alpha
            
            # Bottom edge
            for i in range(blend_height):
                alpha = i / blend_height
                mask[-(i+1), :] *= alpha
            
            # Left edge
            for i in range(blend_width):
                alpha = i / blend_width
                mask[:, i] *= alpha
            
            # Right edge
            for i in range(blend_width):
                alpha = i / blend_width
                mask[:, -(i+1)] *= alpha
            
            # Convert mask to 3 channels
            mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            
            # Apply blending
            result = target_image.copy()
            target_region = result[top:bottom, left:right]
            
            blended_region = (target_region.astype(np.float32) * (1 - mask_3d) + 
                            resized_source.astype(np.float32) * mask_3d)
            
            result[top:bottom, left:right] = blended_region.astype(np.uint8)
            
            return result
            
        except Exception as e:
            logger.error(f"Error blending face boundary: {str(e)}")
            return target_image
    
    def detect_face_orientation(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Detect face orientation (yaw, pitch, roll)
        
        Args:
            landmarks: Facial landmarks
            
        Returns:
            Dictionary with orientation angles
        """
        try:
            if landmarks is None or len(landmarks) < 68:
                return {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}
            
            # Define 3D model points (approximate)
            model_points = np.array([
                (0.0, 0.0, 0.0),             # Nose tip
                (0.0, -330.0, -65.0),        # Chin
                (-225.0, 170.0, -135.0),     # Left eye left corner
                (225.0, 170.0, -135.0),      # Right eye right corner
                (-150.0, -150.0, -125.0),    # Left mouth corner
                (150.0, -150.0, -125.0)      # Right mouth corner
            ])
            
            # 2D image points from landmarks
            image_points = np.array([
                landmarks[30],    # Nose tip
                landmarks[8],     # Chin
                landmarks[36],    # Left eye left corner
                landmarks[45],    # Right eye right corner
                landmarks[48],    # Left mouth corner
                landmarks[54]     # Right mouth corner
            ], dtype=np.float64)
            
            # Camera internals (approximate)
            size = (640, 480)  # Assume standard size
            focal_length = size[1]
            center = (size[1]/2, size[0]/2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)
            
            # Distortion coefficients (assume no distortion)
            dist_coeffs = np.zeros((4, 1))
            
            # Solve PnP
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs
            )
            
            if success:
                # Convert rotation vector to angles
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                
                # Extract Euler angles
                sy = np.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + 
                           rotation_matrix[1, 0] * rotation_matrix[1, 0])
                
                singular = sy < 1e-6
                
                if not singular:
                    x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                    y = np.arctan2(-rotation_matrix[2, 0], sy)
                    z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
                else:
                    x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
                    y = np.arctan2(-rotation_matrix[2, 0], sy)
                    z = 0
                
                # Convert to degrees
                yaw = np.degrees(y)
                pitch = np.degrees(x)
                roll = np.degrees(z)
                
                return {'yaw': yaw, 'pitch': pitch, 'roll': roll}
            
            return {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}
            
        except Exception as e:
            logger.error(f"Error detecting face orientation: {str(e)}")
            return {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model_loaded
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'model_type': 'face_recognition + dlib',
            'detection_model': self.face_detection_model,
            'recognition_tolerance': self.face_recognition_tolerance,
            'dlib_available': self.dlib_available,
            'known_faces_count': len(self.known_faces),
            'model_loaded': self.model_loaded
        }