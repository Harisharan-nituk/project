"""
Utils package for Video Generation App

This package contains utility classes for video processing operations:
- Video loading, processing, and saving
- Human pose estimation and tracking
- Face detection, recognition, and swapping
- Clothing manipulation and replacement
- Background removal and replacement

All utilities are designed to work together for comprehensive video generation.
"""

import logging
from typing import Dict, Any, Optional, List

# Configure logging for utils package
logger = logging.getLogger(__name__)

# Version information
__version__ = "1.0.0"
__author__ = "Video Generation App Team"

# Import all utility classes
try:
    from .video_processor import VideoProcessor
    logger.info("VideoProcessor imported successfully")
except ImportError as e:
    logger.error(f"Failed to import VideoProcessor: {e}")
    VideoProcessor = None

try:
    from .pose_estimator import PoseEstimator
    logger.info("PoseEstimator imported successfully")
except ImportError as e:
    logger.error(f"Failed to import PoseEstimator: {e}")
    PoseEstimator = None

try:
    from .face_swapper import FaceSwapper
    logger.info("FaceSwapper imported successfully")
except ImportError as e:
    logger.error(f"Failed to import FaceSwapper: {e}")
    FaceSwapper = None

try:
    from .clothing_changer import ClothingChanger
    logger.info("ClothingChanger imported successfully")
except ImportError as e:
    logger.error(f"Failed to import ClothingChanger: {e}")
    ClothingChanger = None

try:
    from .background_remover import BackgroundRemover
    logger.info("BackgroundRemover imported successfully")
except ImportError as e:
    logger.error(f"Failed to import BackgroundRemover: {e}")
    BackgroundRemover = None

# Utility registry for easy access
UTILITY_REGISTRY = {
    'video_processor': VideoProcessor,
    'pose_estimator': PoseEstimator,
    'face_swapper': FaceSwapper,
    'clothing_changer': ClothingChanger,
    'background_remover': BackgroundRemover
}

# Available utilities list
AVAILABLE_UTILITIES = [name for name, util_class in UTILITY_REGISTRY.items() if util_class is not None]

def get_utility(util_name: str, *args, **kwargs) -> Optional[Any]:
    """
    Factory function to get utility instances
    
    Args:
        util_name: Name of the utility
        *args: Positional arguments for utility constructor
        **kwargs: Keyword arguments for utility constructor
        
    Returns:
        Utility instance or None if not available
    """
    if util_name not in UTILITY_REGISTRY:
        logger.error(f"Unknown utility: {util_name}. Available utilities: {AVAILABLE_UTILITIES}")
        return None
    
    util_class = UTILITY_REGISTRY[util_name]
    if util_class is None:
        logger.error(f"Utility {util_name} is not available due to import error")
        return None
    
    try:
        util_instance = util_class(*args, **kwargs)
        logger.info(f"Successfully created {util_name} utility instance")
        return util_instance
    except Exception as e:
        logger.error(f"Failed to create {util_name} utility: {e}")
        return None

def get_available_utilities() -> List[str]:
    """
    Get list of available utilities
    
    Returns:
        List of available utility names
    """
    return AVAILABLE_UTILITIES.copy()

def check_utility_availability() -> Dict[str, bool]:
    """
    Check availability of all utilities
    
    Returns:
        Dictionary mapping utility names to availability status
    """
    availability = {}
    
    for util_name, util_class in UTILITY_REGISTRY.items():
        try:
            if util_class is not None:
                # Try to instantiate the utility to check if all dependencies are available
                test_instance = util_class()
                availability[util_name] = True
                logger.info(f"Utility {util_name} is available and working")
            else:
                availability[util_name] = False
                logger.warning(f"Utility {util_name} is not available due to import error")
        except Exception as e:
            availability[util_name] = False
            logger.error(f"Utility {util_name} failed availability check: {e}")
    
    return availability

def initialize_video_pipeline(models_dict: Dict = None) -> Dict[str, Any]:
    """
    Initialize complete video processing pipeline
    
    Args:
        models_dict: Dictionary of pre-initialized models to pass to utilities
        
    Returns:
        Dictionary of initialized utility instances
    """
    logger.info("Initializing video processing pipeline...")
    
    pipeline = {}
    
    # Initialize video processor (no dependencies)
    video_processor = get_utility('video_processor')
    if video_processor:
        pipeline['video_processor'] = video_processor
    
    # Initialize utilities with model dependencies
    if models_dict:
        # Pose estimator
        pose_model = models_dict.get('pose')
        if pose_model:
            pose_estimator = get_utility('pose_estimator', pose_model)
            if pose_estimator:
                pipeline['pose_estimator'] = pose_estimator
        
        # Face swapper
        face_model = models_dict.get('face')
        if face_model:
            face_swapper = get_utility('face_swapper', face_model)
            if face_swapper:
                pipeline['face_swapper'] = face_swapper
        
        # Clothing changer
        segmentation_model = models_dict.get('segmentation')
        if segmentation_model:
            clothing_changer = get_utility('clothing_changer', segmentation_model)
            if clothing_changer:
                pipeline['clothing_changer'] = clothing_changer
    else:
        # Initialize without model dependencies (will use defaults)
        for util_name in ['pose_estimator', 'face_swapper', 'clothing_changer']:
            util_instance = get_utility(util_name)
            if util_instance:
                pipeline[util_name] = util_instance
    
    # Background remover (no model dependency)
    background_remover = get_utility('background_remover')
    if background_remover:
        pipeline['background_remover'] = background_remover
    
    logger.info(f"Video pipeline initialized with {len(pipeline)} utilities")
    return pipeline

def get_pipeline_info(pipeline: Dict[str, Any]) -> Dict[str, Dict]:
    """
    Get information about initialized pipeline
    
    Args:
        pipeline: Initialized pipeline dictionary
        
    Returns:
        Dictionary with pipeline information
    """
    pipeline_info = {}
    
    for util_name, util_instance in pipeline.items():
        try:
            if hasattr(util_instance, 'get_info'):
                pipeline_info[util_name] = util_instance.get_info()
            else:
                pipeline_info[util_name] = {
                    'class': util_instance.__class__.__name__,
                    'available': True,
                    'status': 'initialized'
                }
        except Exception as e:
            pipeline_info[util_name] = {
                'available': False,
                'error': str(e)
            }
    
    return pipeline_info

def process_video_with_pipeline(pipeline: Dict[str, Any], input_path: str, 
                               processing_options: Dict) -> str:
    """
    Process video using the complete pipeline
    
    Args:
        pipeline: Initialized pipeline
        input_path: Path to input video
        processing_options: Dictionary of processing options
        
    Returns:
        Path to processed video
    """
    try:
        logger.info(f"Processing video with pipeline: {input_path}")
        
        # Load video
        video_processor = pipeline.get('video_processor')
        if not video_processor:
            raise RuntimeError("Video processor not available in pipeline")
        
        video_data = video_processor.load_video(input_path)
        frames = video_data['frames']
        fps = video_data['fps']
        
        # Process each frame through the pipeline
        processed_frames = []
        
        for i, frame in enumerate(frames):
            processed_frame = frame.copy()
            
            # Pose estimation
            pose_data = None
            if 'pose_estimator' in pipeline:
                pose_data = pipeline['pose_estimator'].detect_pose(frame)
            
            # Face swapping
            if 'face_swapper' in pipeline and processing_options.get('face_image'):
                processed_frame = pipeline['face_swapper'].swap_face(
                    processed_frame, 
                    processing_options['face_image'], 
                    pose_data
                )
            
            # Clothing change
            if 'clothing_changer' in pipeline and processing_options.get('clothing_style'):
                processed_frame = pipeline['clothing_changer'].change_clothing(
                    processed_frame, 
                    processing_options['clothing_style'], 
                    pose_data or {}
                )
            
            # Background replacement
            if 'background_remover' in pipeline and processing_options.get('background'):
                processed_frame = pipeline['background_remover'].replace_background(
                    processed_frame, 
                    processing_options['background']
                )
            
            processed_frames.append(processed_frame)
            
            # Log progress
            if i % 30 == 0:
                logger.info(f"Processed frame {i}/{len(frames)}")
        
        # Save processed video
        from pathlib import Path
        output_path = str(Path(input_path).parent / f"processed_{Path(input_path).name}")
        
        video_processor.save_video(processed_frames, output_path, fps=fps)
        
        logger.info(f"Video processing completed: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error in pipeline processing: {str(e)}")
        raise

# Utility configuration presets
UTILITY_PRESETS = {
    'fast_processing': {
        'pose_estimator': {
            'model_complexity': 0,
            'min_detection_confidence': 0.3,
            'min_tracking_confidence': 0.3
        },
        'face_swapper': {
            'blend_mode': 'alpha'
        },
        'background_remover': {
            'use_simple_method': True
        }
    },
    'balanced_processing': {
        'pose_estimator': {
            'model_complexity': 1,
            'min_detection_confidence': 0.5,
            'min_tracking_confidence': 0.5
        },
        'face_swapper': {
            'blend_mode': 'seamless'
        },
        'background_remover': {
            'use_simple_method': False
        }
    },
    'high_quality': {
        'pose_estimator': {
            'model_complexity': 2,
            'min_detection_confidence': 0.7,
            'min_tracking_confidence': 0.7
        },
        'face_swapper': {
            'blend_mode': 'seamless',
            'enhance_quality': True
        },
        'background_remover': {
            'use_simple_method': False,
            'apply_refinement': True
        }
    }
}

def initialize_pipeline_with_preset(preset: str = 'balanced_processing', 
                                   models_dict: Dict = None) -> Dict[str, Any]:
    """
    Initialize pipeline with predefined configuration preset
    
    Args:
        preset: Configuration preset name
        models_dict: Dictionary of models to pass to utilities
        
    Returns:
        Dictionary of initialized utility instances
    """
    if preset not in UTILITY_PRESETS:
        logger.warning(f"Unknown preset: {preset}. Using 'balanced_processing' preset.")
        preset = 'balanced_processing'
    
    logger.info(f"Initializing pipeline with '{preset}' preset")
    
    preset_config = UTILITY_PRESETS[preset]
    pipeline = {}
    
    # Initialize video processor (always needed)
    video_processor = get_utility('video_processor')
    if video_processor:
        pipeline['video_processor'] = video_processor
    
    # Initialize other utilities with preset configurations
    for util_name in AVAILABLE_UTILITIES:
        if util_name == 'video_processor':
            continue  # Already initialized
        
        try:
            util_config = preset_config.get(util_name, {})
            
            # Add model dependency if available
            if models_dict:
                if util_name == 'pose_estimator' and 'pose' in models_dict:
                    util_instance = get_utility(util_name, models_dict['pose'], **util_config)
                elif util_name == 'face_swapper' and 'face' in models_dict:
                    util_instance = get_utility(util_name, models_dict['face'], **util_config)
                elif util_name == 'clothing_changer' and 'segmentation' in models_dict:
                    util_instance = get_utility(util_name, models_dict['segmentation'], **util_config)
                else:
                    util_instance = get_utility(util_name, **util_config)
            else:
                util_instance = get_utility(util_name, **util_config)
            
            if util_instance:
                pipeline[util_name] = util_instance
                logger.info(f"Initialized {util_name} with {preset} preset")
            
        except Exception as e:
            logger.error(f"Failed to initialize {util_name} with {preset} preset: {e}")
    
    return pipeline

def validate_processing_options(options: Dict) -> Dict[str, bool]:
    """
    Validate processing options for pipeline
    
    Args:
        options: Processing options dictionary
        
    Returns:
        Dictionary with validation results
    """
    validation = {}
    
    # Check face image
    if 'face_image' in options:
        face_path = options['face_image']
        validation['face_image'] = (
            face_path is not None and 
            os.path.exists(face_path) and 
            face_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        )
    else:
        validation['face_image'] = True  # Optional
    
    # Check clothing style
    if 'clothing_style' in options:
        clothing_style = options['clothing_style']
        validation['clothing_style'] = isinstance(clothing_style, str) and len(clothing_style) > 0
    else:
        validation['clothing_style'] = True  # Optional
    
    # Check background
    if 'background' in options:
        bg_path = options['background']
        validation['background'] = (
            bg_path is not None and 
            os.path.exists(bg_path) and 
            bg_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        )
    else:
        validation['background'] = True  # Optional
    
    # Check text prompt
    if 'text_prompt' in options:
        text_prompt = options['text_prompt']
        validation['text_prompt'] = isinstance(text_prompt, str) and len(text_prompt.strip()) > 0
    else:
        validation['text_prompt'] = True  # Optional
    
    return validation

def create_processing_summary(pipeline: Dict[str, Any], options: Dict) -> Dict:
    """
    Create summary of processing pipeline and options
    
    Args:
        pipeline: Initialized pipeline
        options: Processing options
        
    Returns:
        Processing summary dictionary
    """
    summary = {
        'pipeline_components': list(pipeline.keys()),
        'processing_options': {},
        'estimated_capabilities': {},
        'warnings': []
    }
    
    # Summarize options
    for key, value in options.items():
        if value is not None:
            summary['processing_options'][key] = type(value).__name__
    
    # Estimate capabilities
    summary['estimated_capabilities']['face_swapping'] = (
        'face_swapper' in pipeline and 'face_image' in options
    )
    summary['estimated_capabilities']['clothing_change'] = (
        'clothing_changer' in pipeline and 'clothing_style' in options
    )
    summary['estimated_capabilities']['background_replacement'] = (
        'background_remover' in pipeline and 'background' in options
    )
    summary['estimated_capabilities']['pose_tracking'] = (
        'pose_estimator' in pipeline
    )
    
    # Add warnings
    if 'face_swapper' in pipeline and 'face_image' not in options:
        summary['warnings'].append("Face swapper available but no face image provided")
    
    if 'clothing_changer' in pipeline and 'clothing_style' not in options:
        summary['warnings'].append("Clothing changer available but no clothing style specified")
    
    if 'background_remover' in pipeline and 'background' not in options:
        summary['warnings'].append("Background remover available but no background image provided")
    
    return summary

# Utility helper functions
def extract_video_frames(video_path: str, max_frames: int = None) -> List[np.ndarray]:
    """
    Quick utility to extract frames from video
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to extract
        
    Returns:
        List of video frames
    """
    try:
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frames.append(frame)
            frame_count += 1
            
            if max_frames and frame_count >= max_frames:
                break
        
        cap.release()
        return frames
        
    except Exception as e:
        logger.error(f"Error extracting frames: {e}")
        return []

def save_frames_as_video(frames: List, output_path: str, fps: float = 30.0) -> bool:
    """
    Quick utility to save frames as video
    
    Args:
        frames: List of frames
        output_path: Output video path
        fps: Frames per second
        
    Returns:
        True if successful, False otherwise
    """
    try:
        video_processor = get_utility('video_processor')
        if video_processor:
            video_processor.save_video(frames, output_path, fps=fps)
            return True
        return False
        
    except Exception as e:
        logger.error(f"Error saving video: {e}")
        return False

# Export all public classes and functions
__all__ = [
    'VideoProcessor',
    'PoseEstimator',
    'FaceSwapper',
    'ClothingChanger',
    'BackgroundRemover',
    'get_utility',
    'get_available_utilities',
    'check_utility_availability',
    'initialize_video_pipeline',
    'initialize_pipeline_with_preset',
    'validate_processing_options',
    'create_processing_summary',
    'extract_video_frames',
    'save_frames_as_video',
    'UTILITY_REGISTRY',
    'AVAILABLE_UTILITIES',
    'UTILITY_PRESETS'
]

# Package initialization logging
logger.info(f"Utils package initialized. Available utilities: {AVAILABLE_UTILITIES}")
logger.info(f"Package version: {__version__}")

# Dependency check for utils package
def _check_utils_dependencies():
    """Check if critical dependencies for utils are available"""
    dependencies_status = {}
    
    # Check OpenCV
    try:
        import cv2
        dependencies_status['opencv'] = cv2.__version__
    except ImportError:
        dependencies_status['opencv'] = 'Not available'
        logger.error("OpenCV not available - video processing will not work")
    
    # Check NumPy
    try:
        import numpy as np
        dependencies_status['numpy'] = np.__version__
    except ImportError:
        dependencies_status['numpy'] = 'Not available'
        logger.error("NumPy not available - array operations will not work")
    
    # Check MediaPipe
    try:
        import mediapipe as mp
        dependencies_status['mediapipe'] = mp.__version__
    except ImportError:
        dependencies_status['mediapipe'] = 'Not available'
        logger.warning("MediaPipe not available - pose estimation will use fallback")
    
    # Check face_recognition
    try:
        import face_recognition
        dependencies_status['face_recognition'] = 'Available'
    except ImportError:
        dependencies_status['face_recognition'] = 'Not available'
        logger.warning("face_recognition not available - face swapping will be limited")
    
    # Check dlib
    try:
        import dlib
        dependencies_status['dlib'] = 'Available'
    except ImportError:
        dependencies_status['dlib'] = 'Not available'
        logger.warning("dlib not available - advanced face processing will be limited")
    
    # Check rembg
    try:
        import rembg
        dependencies_status['rembg'] = 'Available'
    except ImportError:
        dependencies_status['rembg'] = 'Not available'
        logger.info("rembg not available - will use basic background removal")
    
    # Check moviepy
    try:
        import moviepy
        dependencies_status['moviepy'] = moviepy.__version__
    except ImportError:
        dependencies_status['moviepy'] = 'Not available'
        logger.warning("moviepy not available - some video operations will be limited")
    
    return dependencies_status

# Run dependency check on import
UTILS_DEPENDENCIES_STATUS = _check_utils_dependencies()

# Log dependency status
logger.info("Utils dependency check completed:")
for dep, status in UTILS_DEPENDENCIES_STATUS.items():
    logger.info(f"  {dep}: {status}")

# Critical dependency warnings
critical_deps = ['opencv', 'numpy']
missing_critical = [dep for dep in critical_deps if UTILS_DEPENDENCIES_STATUS.get(dep) == 'Not available']

if missing_critical:
    logger.error(f"Critical dependencies missing: {missing_critical}")
    logger.error("Utils package may not function properly")
else:
    logger.info("All critical dependencies available")

# Optional dependency info
optional_deps = ['mediapipe', 'face_recognition', 'dlib', 'rembg', 'moviepy']
missing_optional = [dep for dep in optional_deps if UTILS_DEPENDENCIES_STATUS.get(dep) == 'Not available']

if missing_optional:
    logger.info(f"Optional dependencies missing: {missing_optional}")
    logger.info("Some advanced features may not be available")