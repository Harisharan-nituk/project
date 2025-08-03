"""
Models package for Video Generation App

This package contains all the machine learning models used for:
- Pose estimation and tracking
- Face detection and recognition
- Image segmentation and human parsing

All models are designed to work together for comprehensive video processing.
"""

import logging
from typing import Dict, Any, Optional

# Configure logging for models package
logger = logging.getLogger(__name__)

# Version information
__version__ = "1.0.0"
__author__ = "Video Generation App Team"

# Import all model classes
try:
    from .pose_model import PoseModel
    logger.info("PoseModel imported successfully")
except ImportError as e:
    logger.error(f"Failed to import PoseModel: {e}")
    PoseModel = None

try:
    from .face_model import FaceModel
    logger.info("FaceModel imported successfully")
except ImportError as e:
    logger.error(f"Failed to import FaceModel: {e}")
    FaceModel = None

try:
    from .segmentation_model import SegmentationModel
    logger.info("SegmentationModel imported successfully")
except ImportError as e:
    logger.error(f"Failed to import SegmentationModel: {e}")
    SegmentationModel = None

# Model registry for easy access
MODEL_REGISTRY = {
    'pose': PoseModel,
    'face': FaceModel,
    'segmentation': SegmentationModel
}

# Available models list
AVAILABLE_MODELS = [name for name, model_class in MODEL_REGISTRY.items() if model_class is not None]

def get_model(model_name: str, **kwargs) -> Optional[Any]:
    """
    Factory function to get model instances
    
    Args:
        model_name: Name of the model ('pose', 'face', 'segmentation')
        **kwargs: Additional arguments to pass to model constructor
        
    Returns:
        Model instance or None if not available
    """
    if model_name not in MODEL_REGISTRY:
        logger.error(f"Unknown model: {model_name}. Available models: {AVAILABLE_MODELS}")
        return None
    
    model_class = MODEL_REGISTRY[model_name]
    if model_class is None:
        logger.error(f"Model {model_name} is not available due to import error")
        return None
    
    try:
        model_instance = model_class(**kwargs)
        logger.info(f"Successfully created {model_name} model instance")
        return model_instance
    except Exception as e:
        logger.error(f"Failed to create {model_name} model: {e}")
        return None

def get_available_models() -> list:
    """
    Get list of available models
    
    Returns:
        List of available model names
    """
    return AVAILABLE_MODELS.copy()

def check_model_availability() -> Dict[str, bool]:
    """
    Check availability of all models
    
    Returns:
        Dictionary mapping model names to availability status
    """
    availability = {}
    
    for model_name, model_class in MODEL_REGISTRY.items():
        try:
            if model_class is not None:
                # Try to instantiate the model to check if all dependencies are available
                test_instance = model_class()
                availability[model_name] = True
                logger.info(f"Model {model_name} is available and working")
            else:
                availability[model_name] = False
                logger.warning(f"Model {model_name} is not available due to import error")
        except Exception as e:
            availability[model_name] = False
            logger.error(f"Model {model_name} failed availability check: {e}")
    
    return availability

def get_model_info() -> Dict[str, Dict]:
    """
    Get information about all available models
    
    Returns:
        Dictionary with model information
    """
    model_info = {}
    
    for model_name in AVAILABLE_MODELS:
        try:
            model_instance = get_model(model_name)
            if model_instance and hasattr(model_instance, 'get_model_info'):
                model_info[model_name] = model_instance.get_model_info()
            else:
                model_info[model_name] = {
                    'available': False,
                    'error': 'Model instance could not be created or lacks info method'
                }
        except Exception as e:
            model_info[model_name] = {
                'available': False,
                'error': str(e)
            }
    
    return model_info

def initialize_all_models(**kwargs) -> Dict[str, Any]:
    """
    Initialize all available models
    
    Args:
        **kwargs: Common arguments to pass to all models
        
    Returns:
        Dictionary of initialized model instances
    """
    initialized_models = {}
    
    logger.info("Initializing all available models...")
    
    for model_name in AVAILABLE_MODELS:
        try:
            model_instance = get_model(model_name, **kwargs)
            if model_instance:
                initialized_models[model_name] = model_instance
                logger.info(f"Successfully initialized {model_name} model")
            else:
                logger.warning(f"Failed to initialize {model_name} model")
        except Exception as e:
            logger.error(f"Error initializing {model_name} model: {e}")
    
    logger.info(f"Initialized {len(initialized_models)} out of {len(AVAILABLE_MODELS)} models")
    return initialized_models

# Model configuration presets
MODEL_PRESETS = {
    'fast': {
        'pose': {'model_complexity': 0, 'min_detection_confidence': 0.3},
        'face': {'face_detection_model': 'hog'},
        'segmentation': {'model_name': 'fcn_resnet50'}
    },
    'balanced': {
        'pose': {'model_complexity': 1, 'min_detection_confidence': 0.5},
        'face': {'face_detection_model': 'hog'},
        'segmentation': {'model_name': 'deeplabv3_resnet50'}
    },
    'accurate': {
        'pose': {'model_complexity': 2, 'min_detection_confidence': 0.7},
        'face': {'face_detection_model': 'cnn'},
        'segmentation': {'model_name': 'deeplabv3_resnet101'}
    }
}

def initialize_models_with_preset(preset: str = 'balanced') -> Dict[str, Any]:
    """
    Initialize models with predefined configuration preset
    
    Args:
        preset: Configuration preset ('fast', 'balanced', 'accurate')
        
    Returns:
        Dictionary of initialized model instances
    """
    if preset not in MODEL_PRESETS:
        logger.warning(f"Unknown preset: {preset}. Using 'balanced' preset.")
        preset = 'balanced'
    
    logger.info(f"Initializing models with '{preset}' preset")
    
    preset_config = MODEL_PRESETS[preset]
    initialized_models = {}
    
    for model_name in AVAILABLE_MODELS:
        try:
            model_config = preset_config.get(model_name, {})
            model_instance = get_model(model_name, **model_config)
            
            if model_instance:
                initialized_models[model_name] = model_instance
                logger.info(f"Successfully initialized {model_name} model with {preset} preset")
            else:
                logger.warning(f"Failed to initialize {model_name} model with {preset} preset")
                
        except Exception as e:
            logger.error(f"Error initializing {model_name} model with {preset} preset: {e}")
    
    return initialized_models

# Export all public classes and functions
__all__ = [
    'PoseModel',
    'FaceModel', 
    'SegmentationModel',
    'get_model',
    'get_available_models',
    'check_model_availability',
    'get_model_info',
    'initialize_all_models',
    'initialize_models_with_preset',
    'MODEL_REGISTRY',
    'AVAILABLE_MODELS',
    'MODEL_PRESETS'
]

# Package initialization logging
logger.info(f"Models package initialized. Available models: {AVAILABLE_MODELS}")
logger.info(f"Package version: {__version__}")

# Verify critical dependencies
def _check_dependencies():
    """Check if critical dependencies are available"""
    dependencies_status = {}
    
    # Check OpenCV
    try:
        import cv2
        dependencies_status['opencv'] = cv2.__version__
    except ImportError:
        dependencies_status['opencv'] = 'Not available'
        logger.error("OpenCV not available")
    
    # Check MediaPipe
    try:
        import mediapipe as mp
        dependencies_status['mediapipe'] = mp.__version__
    except ImportError:
        dependencies_status['mediapipe'] = 'Not available'
        logger.error("MediaPipe not available")
    
    # Check PyTorch
    try:
        import torch
        dependencies_status['torch'] = torch.__version__
        dependencies_status['cuda_available'] = torch.cuda.is_available()
    except ImportError:
        dependencies_status['torch'] = 'Not available'
        dependencies_status['cuda_available'] = False
        logger.error("PyTorch not available")
    
    # Check face_recognition
    try:
        import face_recognition
        dependencies_status['face_recognition'] = 'Available'
    except ImportError:
        dependencies_status['face_recognition'] = 'Not available'
        logger.error("face_recognition not available")
    
    return dependencies_status

# Run dependency check on import
DEPENDENCIES_STATUS = _check_dependencies()

# Log dependency status
logger.info("Dependency check completed:")
for dep, status in DEPENDENCIES_STATUS.items():
    logger.info(f"  {dep}: {status}")

# Warn about missing critical dependencies
if DEPENDENCIES_STATUS.get('opencv') == 'Not available':
    logger.warning("OpenCV is not available - video processing will be limited")

if DEPENDENCIES_STATUS.get('torch') == 'Not available':
    logger.warning("PyTorch is not available - segmentation model will not work")

if not DEPENDENCIES_STATUS.get('cuda_available', False):
    logger.info("CUDA not available - models will run on CPU")