import axios from 'axios';

// Create axios instance with default config
const api = axios.create({
  baseURL: process.env.REACT_APP_API_BASE_URL || '/api',
  timeout: 300000, // 5 minutes for video processing
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('authToken');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    
    // Log requests in development
    if (process.env.NODE_ENV === 'development') {
      console.log('API Request:', config.method?.toUpperCase(), config.url);
    }
    
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => {
    // Log responses in development
    if (process.env.NODE_ENV === 'development') {
      console.log('API Response:', response.status, response.config.url);
    }
    
    return response;
  },
  (error) => {
    // Handle common errors
    if (error.response?.status === 401) {
      // Unauthorized - redirect to login
      localStorage.removeItem('authToken');
      window.location.href = '/login';
    }
    
    if (error.response?.status === 403) {
      // Forbidden
      console.error('Access forbidden');
    }
    
    if (error.response?.status >= 500) {
      // Server error
      console.error('Server error:', error.response.data);
    }
    
    // Log errors in development
    if (process.env.NODE_ENV === 'development') {
      console.error('API Error:', error.response?.status, error.config?.url, error.message);
    }
    
    return Promise.reject(error);
  }
);

export const apiService = {
  // System endpoints
  async getSystemInfo() {
    try {
      const response = await api.get('/system/info');
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.error || 'Failed to get system info');
    }
  },

  async getHealthCheck() {
    try {
      const response = await api.get('/health');
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.error || 'Health check failed');
    }
  },

  // File upload endpoints
  async uploadFile(formData, onUploadProgress) {
    try {
      const response = await api.post('/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          if (onUploadProgress) {
            onUploadProgress(progressEvent);
          }
        },
      });
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.error || 'Upload failed');
    }
  },

  async getUploadedFiles() {
    try {
      const response = await api.get('/files');
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.error || 'Failed to get uploaded files');
    }
  },

  async deleteFile(fileId) {
    try {
      const response = await api.delete(`/files/${fileId}`);
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.error || 'Failed to delete file');
    }
  },

  // Processing endpoints
  async startProcessing(fileId, options = {}) {
    try {
      const response = await api.post('/process', {
        file_id: fileId,
        ...options,
      });
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.error || 'Failed to start processing');
    }
  },

  async getProcessingStatus(taskId) {
    try {
      const response = await api.get(`/status/${taskId}`);
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.error || 'Failed to get processing status');
    }
  },

  async cancelProcessing(taskId) {
    try {
      const response = await api.post(`/cancel/${taskId}`);
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.error || 'Failed to cancel processing');
    }
  },

  async downloadResult(taskId) {
    try {
      const response = await api.get(`/download/${taskId}`, {
        responseType: 'blob',
      });
      
      // Create download URL
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      
      // Get filename from headers or use default
      const contentDisposition = response.headers['content-disposition'];
      let filename = 'processed_video.mp4';
      
      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename="(.+)"/);
        if (filenameMatch) {
          filename = filenameMatch[1];
        }
      }
      
      link.setAttribute('download', filename);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
      
      return { success: true, filename };
    } catch (error) {
      throw new Error(error.response?.data?.error || 'Failed to download result');
    }
  },

  // Model and configuration endpoints
  async getAvailableModels() {
    try {
      const response = await api.get('/models');
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.error || 'Failed to get available models');
    }
  },

  async getClothingStyles() {
    try {
      const response = await api.get('/clothing/styles');
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.error || 'Failed to get clothing styles');
    }
  },

  async getBackgrounds() {
    try {
      const response = await api.get('/backgrounds');
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.error || 'Failed to get backgrounds');
    }
  },

  // Utility endpoints
  async generateThumbnail(fileId) {
    try {
      const response = await api.post(`/thumbnail/${fileId}`);
      return response.data.thumbnail_url;
    } catch (error) {
      throw new Error(error.response?.data?.error || 'Failed to generate thumbnail');
    }
  },

  async cleanupFiles() {
    try {
      const response = await api.post('/cleanup');
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.error || 'Failed to cleanup files');
    }
  },

  // Projects endpoints
  async getProjects() {
    try {
      const response = await api.get('/projects');
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.error || 'Failed to get projects');
    }
  },

  async createProject(projectData) {
    try {
      const response = await api.post('/projects', projectData);
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.error || 'Failed to create project');
    }
  },

  async getProject(projectId) {
    try {
      const response = await api.get(`/projects/${projectId}`);
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.error || 'Failed to get project');
    }
  },

  async updateProject(projectId, updates) {
    try {
      const response = await api.put(`/projects/${projectId}`, updates);
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.error || 'Failed to update project');
    }
  },

  async deleteProject(projectId) {
    try {
      const response = await api.delete(`/projects/${projectId}`);
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.error || 'Failed to delete project');
    }
  },

  // Statistics endpoints
  async getStatistics() {
    try {
      const response = await api.get('/stats');
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.error || 'Failed to get statistics');
    }
  },

  async getProcessingHistory(limit = 50) {
    try {
      const response = await api.get(`/history?limit=${limit}`);
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.error || 'Failed to get processing history');
    }
  },

  // Settings endpoints
  async getUserSettings() {
    try {
      const response = await api.get('/settings');
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.error || 'Failed to get user settings');
    }
  },

  async updateUserSettings(settings) {
    try {
      const response = await api.put('/settings', settings);
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.error || 'Failed to update user settings');
    }
  },

  // Video analysis endpoints
  async analyzeVideo(fileId) {
    try {
      const response = await api.post(`/analyze/${fileId}`);
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.error || 'Failed to analyze video');
    }
  },

  async getVideoInfo(fileId) {
    try {
      const response = await api.get(`/video-info/${fileId}`);
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.error || 'Failed to get video info');
    }
  },

  // Template endpoints
  async getTemplates() {
    try {
      const response = await api.get('/templates');
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.error || 'Failed to get templates');
    }
  },

  async createTemplate(templateData) {
    try {
      const response = await api.post('/templates', templateData);
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.error || 'Failed to create template');
    }
  },

  async applyTemplate(templateId, fileId) {
    try {
      const response = await api.post(`/templates/${templateId}/apply`, {
        file_id: fileId,
      });
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.error || 'Failed to apply template');
    }
  },

  // Batch processing endpoints
  async startBatchProcessing(fileIds, options = {}) {
    try {
      const response = await api.post('/batch/process', {
        file_ids: fileIds,
        ...options,
      });
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.error || 'Failed to start batch processing');
    }
  },

  async getBatchStatus(batchId) {
    try {
      const response = await api.get(`/batch/status/${batchId}`);
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.error || 'Failed to get batch status');
    }
  },

  // Quality assessment endpoints
  async assessVideoQuality(fileId) {
    try {
      const response = await api.post(`/quality/assess/${fileId}`);
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.error || 'Failed to assess video quality');
    }
  },

  async enhanceVideoQuality(fileId, enhancementOptions = {}) {
    try {
      const response = await api.post(`/quality/enhance/${fileId}`, enhancementOptions);
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.error || 'Failed to enhance video quality');
    }
  },

  // Export endpoints
  async exportProject(projectId, format = 'mp4') {
    try {
      const response = await api.post(`/export/project/${projectId}`, { format });
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.error || 'Failed to export project');
    }
  },

  async getExportStatus(exportId) {
    try {
      const response = await api.get(`/export/status/${exportId}`);
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.error || 'Failed to get export status');
    }
  },

  // Share endpoints
  async shareVideo(fileId, shareOptions = {}) {
    try {
      const response = await api.post(`/share/${fileId}`, shareOptions);
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.error || 'Failed to share video');
    }
  },

  async getSharedVideo(shareToken) {
    try {
      const response = await api.get(`/shared/${shareToken}`);
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.error || 'Failed to get shared video');
    }
  },

  // Utility functions
  createFormData(data) {
    const formData = new FormData();
    
    Object.keys(data).forEach(key => {
      const value = data[key];
      
      if (value instanceof File || value instanceof Blob) {
        formData.append(key, value);
      } else if (typeof value === 'object' && value !== null) {
        formData.append(key, JSON.stringify(value));
      } else {
        formData.append(key, value);
      }
    });
    
    return formData;
  },

  async uploadWithProgress(endpoint, data, onProgress) {
    const formData = this.createFormData(data);
    
    return api.post(endpoint, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: onProgress,
    });
  },

  // Error handling utilities
  handleApiError(error) {
    if (error.response) {
      // Server responded with error status
      const { status, data } = error.response;
      
      switch (status) {
        case 400:
          return { type: 'validation', message: data.error || 'Invalid request' };
        case 401:
          return { type: 'auth', message: 'Authentication required' };
        case 403:
          return { type: 'permission', message: 'Access denied' };
        case 404:
          return { type: 'notfound', message: 'Resource not found' };
        case 413:
          return { type: 'filesize', message: 'File too large' };
        case 429:
          return { type: 'ratelimit', message: 'Too many requests' };
        case 500:
          return { type: 'server', message: 'Server error' };
        default:
          return { type: 'unknown', message: data.error || 'Unknown error' };
      }
    } else if (error.request) {
      // Network error
      return { type: 'network', message: 'Network error - please check your connection' };
    } else {
      // Other error
      return { type: 'client', message: error.message || 'Client error' };
    }
  },

  // Retry logic for failed requests
  async retryRequest(requestFn, maxRetries = 3, delay = 1000) {
    let lastError;
    
    for (let i = 0; i < maxRetries; i++) {
      try {
        return await requestFn();
      } catch (error) {
        lastError = error;
        
        // Don't retry on certain error types
        if (error.response?.status === 401 || error.response?.status === 403) {
          throw error;
        }
        
        // Wait before retrying
        if (i < maxRetries - 1) {
          await new Promise(resolve => setTimeout(resolve, delay * Math.pow(2, i)));
        }
      }
    }
    
    throw lastError;
  },
};

// Export default instance
export default apiService;

// Named exports for convenience
export const {
  getSystemInfo,
  uploadFile,
  startProcessing,
  getProcessingStatus,
  downloadResult,
  getClothingStyles,
  getBackgrounds,
} = apiService;