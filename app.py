#!/usr/bin/env python3
"""
Flask Web Application for Video Generation
"""

import os
import sys
import logging
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from werkzeug.utils import secure_filename
import uuid
import json
from datetime import datetime
import threading
import queue
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import *
from main import VideoGenerator

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG['level']),
    format=LOGGING_CONFIG['format']
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = WEB_CONFIG['max_file_size']

# Global video generator instance
video_generator = None
processing_queue = queue.Queue()
processing_results = {}

def initialize_video_generator():
    """Initialize video generator in background"""
    global video_generator
    try:
        logger.info("Initializing video generator...")
        video_generator = VideoGenerator()
        logger.info("Video generator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize video generator: {str(e)}")
        video_generator = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in WEB_CONFIG['allowed_extensions']

def process_video_task(task_id, input_path, options):
    """Background task for video processing"""
    global processing_results, video_generator
    
    try:
        logger.info(f"Starting video processing task: {task_id}")
        
        if video_generator is None:
            raise RuntimeError("Video generator not initialized")
        
        # Update processing status
        processing_results[task_id] = {
            'status': 'processing',
            'progress': 0,
            'message': 'Processing video...',
            'start_time': datetime.now().isoformat()
        }
        
        # Process video
        output_path = video_generator.process_video(
            input_path=input_path,
            face_image=options.get('face_image'),
            clothing_style=options.get('clothing_style'),
            background=options.get('background'),
            text_prompt=options.get('text_prompt')
        )
        
        # Update results
        processing_results[task_id] = {
            'status': 'completed',
            'progress': 100,
            'message': 'Video processing completed successfully',
            'output_path': output_path,
            'end_time': datetime.now().isoformat()
        }
        
        logger.info(f"Video processing task completed: {task_id}")
        
    except Exception as e:
        logger.error(f"Error in video processing task {task_id}: {str(e)}")
        processing_results[task_id] = {
            'status': 'error',
            'progress': 0,
            'message': f'Error: {str(e)}',
            'end_time': datetime.now().isoformat()
        }

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload')
def upload_page():
    """Upload page"""
    return render_template('upload.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Generate unique filename
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = INPUT_VIDEOS_DIR / unique_filename
        
        # Save uploaded file
        file.save(str(file_path))
        
        logger.info(f"File uploaded successfully: {file_path}")
        
        return jsonify({
            'success': True,
            'file_id': unique_filename,
            'filename': filename,
            'file_path': str(file_path)
        })
        
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/process', methods=['POST'])
def process_video():
    """Start video processing"""
    try:
        data = request.json
        
        if not data or 'file_id' not in data:
            return jsonify({'error': 'No file ID provided'}), 400
        
        file_id = data['file_id']
        input_path = INPUT_VIDEOS_DIR / file_id
        
        if not input_path.exists():
            return jsonify({'error': 'Input file not found'}), 404
        
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Extract processing options
        options = {
            'face_image': data.get('face_image'),
            'clothing_style': data.get('clothing_style'),
            'background': data.get('background'),
            'text_prompt': data.get('text_prompt')
        }
        
        # Handle face image upload
        if 'face_image_file' in request.files:
            face_file = request.files['face_image_file']
            if face_file and allowed_file(face_file.filename):
                face_filename = f"{uuid.uuid4()}_{secure_filename(face_file.filename)}"
                face_path = FACES_DIR / face_filename
                face_file.save(str(face_path))
                options['face_image'] = str(face_path)
        
        # Handle background image upload
        if 'background_file' in request.files:
            bg_file = request.files['background_file']
            if bg_file and allowed_file(bg_file.filename):
                bg_filename = f"{uuid.uuid4()}_{secure_filename(bg_file.filename)}"
                bg_path = BACKGROUNDS_DIR / bg_filename
                bg_file.save(str(bg_path))
                options['background'] = str(bg_path)
        
        # Start background processing
        thread = threading.Thread(
            target=process_video_task,
            args=(task_id, str(input_path), options)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'message': 'Video processing started'
        })
        
    except Exception as e:
        logger.error(f"Error starting video processing: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/status/<task_id>')
def get_processing_status(task_id):
    """Get processing status"""
    try:
        if task_id in processing_results:
            return jsonify(processing_results[task_id])
        else:
            return jsonify({
                'status': 'not_found',
                'message': 'Task not found'
            }), 404
            
    except Exception as e:
        logger.error(f"Error getting processing status: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/download/<task_id>')
def download_result(task_id):
    """Download processed video"""
    try:
        if task_id not in processing_results:
            return jsonify({'error': 'Task not found'}), 404
        
        result = processing_results[task_id]
        
        if result['status'] != 'completed':
            return jsonify({'error': 'Processing not completed'}), 400
        
        output_path = result.get('output_path')
        
        if not output_path or not os.path.exists(output_path):
            return jsonify({'error': 'Output file not found'}), 404
        
        return send_file(
            output_path,
            as_attachment=True,
            download_name=f"generated_video_{task_id}.mp4"
        )
        
    except Exception as e:
        logger.error(f"Error downloading result: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/clothing/styles')
def get_clothing_styles():
    """Get available clothing styles"""
    try:
        if video_generator and hasattr(video_generator.clothing_changer, 'get_available_styles'):
            styles = video_generator.clothing_changer.get_available_styles()
        else:
            # Default styles
            styles = [
                'formal_suit', 'casual_wear', 'dress', 'business_suit',
                'shirt', 'jacket', 'summer_wear', 'winter_wear'
            ]
        
        return jsonify({
            'success': True,
            'styles': styles
        })
        
    except Exception as e:
        logger.error(f"Error getting clothing styles: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/backgrounds')
def get_backgrounds():
    """Get available background images"""
    try:
        background_files = []
        
        if BACKGROUNDS_DIR.exists():
            for file_path in BACKGROUNDS_DIR.glob('*'):
                if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    background_files.append({
                        'name': file_path.name,
                        'path': str(file_path)
                    })
        
        return jsonify({
            'success': True,
            'backgrounds': background_files
        })
        
    except Exception as e:
        logger.error(f"Error getting backgrounds: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'video_generator_ready': video_generator is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/system/info')
def system_info():
    """Get system information"""
    try:
        import torch
        import cv2
        
        info = {
            'python_version': sys.version,
            'opencv_version': cv2.__version__,
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'video_generator_status': 'initialized' if video_generator else 'not_initialized'
        }
        
        if torch.cuda.is_available():
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory'] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"Error getting system info: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/cleanup')
def cleanup_files():
    """Clean up old files"""
    try:
        cleaned_files = 0
        current_time = time.time()
        max_age = 24 * 60 * 60  # 24 hours
        
        # Clean up input videos
        for file_path in INPUT_VIDEOS_DIR.glob('*'):
            if file_path.is_file() and current_time - file_path.stat().st_mtime > max_age:
                file_path.unlink()
                cleaned_files += 1
        
        # Clean up output videos
        for file_path in OUTPUT_VIDEOS_DIR.glob('*'):
            if file_path.is_file() and current_time - file_path.stat().st_mtime > max_age:
                file_path.unlink()
                cleaned_files += 1
        
        # Clean up temporary files
        for file_path in TEMP_DIR.glob('*'):
            if file_path.is_file() and current_time - file_path.stat().st_mtime > max_age:
                file_path.unlink()
                cleaned_files += 1
        
        return jsonify({
            'success': True,
            'cleaned_files': cleaned_files,
            'message': f'Cleaned up {cleaned_files} old files'
        })
        
    except Exception as e:
        logger.error(f"Error cleaning up files: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

def create_app():
    """Application factory"""
    # Initialize video generator in background
    init_thread = threading.Thread(target=initialize_video_generator)
    init_thread.daemon = True
    init_thread.start()
    
    return app

if __name__ == '__main__':
    # Create Flask app
    app = create_app()
    
    logger.info("Starting Flask application...")
    logger.info(f"Server will run on {WEB_CONFIG['host']}:{WEB_CONFIG['port']}")
    
    # Run Flask app
    app.run(
        host=WEB_CONFIG['host'],
        port=WEB_CONFIG['port'],
        debug=WEB_CONFIG['debug'],
        threaded=True
    )