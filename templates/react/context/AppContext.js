
import React, { createContext, useContext, useReducer, useEffect } from 'react';

// Initial state
const initialState = {
  // UI State
  isLoading: false,
  error: null,
  sidebarOpen: false,
  
  // System State
  systemInfo: null,
  modelStatus: {},
  
  // Video Processing State
  currentProject: null,
  processingQueue: [],
  processingHistory: [],
  
  // Upload State
  uploadProgress: 0,
  uploadedFiles: [],
  
  // Settings
  userPreferences: {
    theme: 'dark',
    autoSave: true,
    processOnUpload: false,
    defaultOutputFormat: 'mp4',
    defaultQuality: 'high',
  },
  
  // Processing Options
  processingOptions: {
    faceImage: null,
    clothingStyle: null,
    background: null,
    textPrompt: '',
  },
};

// Action types
const actionTypes = {
  // UI Actions
  SET_LOADING: 'SET_LOADING',
  SET_ERROR: 'SET_ERROR',
  TOGGLE_SIDEBAR: 'TOGGLE_SIDEBAR',
  SET_SIDEBAR_OPEN: 'SET_SIDEBAR_OPEN',
  
  // System Actions
  SET_SYSTEM_INFO: 'SET_SYSTEM_INFO',
  SET_MODEL_STATUS: 'SET_MODEL_STATUS',
  
  // Project Actions
  SET_CURRENT_PROJECT: 'SET_CURRENT_PROJECT',
  CREATE_PROJECT: 'CREATE_PROJECT',
  UPDATE_PROJECT: 'UPDATE_PROJECT',
  
  // Processing Actions
  ADD_TO_QUEUE: 'ADD_TO_QUEUE',
  REMOVE_FROM_QUEUE: 'REMOVE_FROM_QUEUE',
  UPDATE_QUEUE_ITEM: 'UPDATE_QUEUE_ITEM',
  ADD_TO_HISTORY: 'ADD_TO_HISTORY',
  CLEAR_HISTORY: 'CLEAR_HISTORY',
  
  // Upload Actions
  SET_UPLOAD_PROGRESS: 'SET_UPLOAD_PROGRESS',
  ADD_UPLOADED_FILE: 'ADD_UPLOADED_FILE',
  REMOVE_UPLOADED_FILE: 'REMOVE_UPLOADED_FILE',
  CLEAR_UPLOADED_FILES: 'CLEAR_UPLOADED_FILES',
  
  // Settings Actions
  UPDATE_PREFERENCES: 'UPDATE_PREFERENCES',
  UPDATE_PROCESSING_OPTIONS: 'UPDATE_PROCESSING_OPTIONS',
  RESET_PROCESSING_OPTIONS: 'RESET_PROCESSING_OPTIONS',
};

// Reducer function
function appReducer(state, action) {
  switch (action.type) {
    case actionTypes.SET_LOADING:
      return { ...state, isLoading: action.payload };
    
    case actionTypes.SET_ERROR:
      return { ...state, error: action.payload };
    
    case actionTypes.TOGGLE_SIDEBAR:
      return { ...state, sidebarOpen: !state.sidebarOpen };
    
    case actionTypes.SET_SIDEBAR_OPEN:
      return { ...state, sidebarOpen: action.payload };
    
    case actionTypes.SET_SYSTEM_INFO:
      return { ...state, systemInfo: action.payload };
    
    case actionTypes.SET_MODEL_STATUS:
      return { ...state, modelStatus: action.payload };
    
    case actionTypes.SET_CURRENT_PROJECT:
      return { ...state, currentProject: action.payload };
    
    case actionTypes.CREATE_PROJECT:
      const newProject = {
        id: Date.now().toString(),
        name: action.payload.name || 'Unnamed Project',
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        files: [],
        processingOptions: { ...state.processingOptions },
        status: 'created',
        ...action.payload,
      };
      return {
        ...state,
        currentProject: newProject,
      };
    
    case actionTypes.UPDATE_PROJECT:
      if (!state.currentProject) return state;
      return {
        ...state,
        currentProject: {
          ...state.currentProject,
          ...action.payload,
          updatedAt: new Date().toISOString(),
        },
      };
    
    case actionTypes.ADD_TO_QUEUE:
      return {
        ...state,
        processingQueue: [...state.processingQueue, action.payload],
      };
    
    case actionTypes.REMOVE_FROM_QUEUE:
      return {
        ...state,
        processingQueue: state.processingQueue.filter(
          item => item.id !== action.payload
        ),
      };
    
    case actionTypes.UPDATE_QUEUE_ITEM:
      return {
        ...state,
        processingQueue: state.processingQueue.map(item =>
          item.id === action.payload.id
            ? { ...item, ...action.payload.updates }
            : item
        ),
      };
    
    case actionTypes.ADD_TO_HISTORY:
      return {
        ...state,
        processingHistory: [action.payload, ...state.processingHistory],
      };
    
    case actionTypes.CLEAR_HISTORY:
      return {
        ...state,
        processingHistory: [],
      };
    
    case actionTypes.SET_UPLOAD_PROGRESS:
      return { ...state, uploadProgress: action.payload };
    
    case actionTypes.ADD_UPLOADED_FILE:
      return {
        ...state,
        uploadedFiles: [...state.uploadedFiles, action.payload],
      };
    
    case actionTypes.REMOVE_UPLOADED_FILE:
      return {
        ...state,
        uploadedFiles: state.uploadedFiles.filter(
          file => file.id !== action.payload
        ),
      };
    
    case actionTypes.CLEAR_UPLOADED_FILES:
      return { ...state, uploadedFiles: [] };
    
    case actionTypes.UPDATE_PREFERENCES:
      return {
        ...state,
        userPreferences: { ...state.userPreferences, ...action.payload },
      };
    
    case actionTypes.UPDATE_PROCESSING_OPTIONS:
      return {
        ...state,
        processingOptions: { ...state.processingOptions, ...action.payload },
      };
    
    case actionTypes.RESET_PROCESSING_OPTIONS:
      return {
        ...state,
        processingOptions: initialState.processingOptions,
      };
    
    default:
      return state;
  }
}

// Create context
const AppContext = createContext();

// Custom hook to use the context
export const useAppContext = () => {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useAppContext must be used within an AppProvider');
  }
  return context;
};

// Context provider component
export const AppProvider = ({ children }) => {
  const [state, dispatch] = useReducer(appReducer, initialState);

  // Load preferences from localStorage on mount
  useEffect(() => {
    const savedPreferences = localStorage.getItem('videoGenPreferences');
    if (savedPreferences) {
      try {
        const preferences = JSON.parse(savedPreferences);
        dispatch({
          type: actionTypes.UPDATE_PREFERENCES,
          payload: preferences,
        });
      } catch (error) {
        console.error('Failed to load preferences:', error);
      }
    }
  }, []);

  // Save preferences to localStorage when they change
  useEffect(() => {
    localStorage.setItem(
      'videoGenPreferences',
      JSON.stringify(state.userPreferences)
    );
  }, [state.userPreferences]);

  // Action creators
  const actions = {
    // UI Actions
    setLoading: (loading) => dispatch({
      type: actionTypes.SET_LOADING,
      payload: loading,
    }),
    
    setError: (error) => dispatch({
      type: actionTypes.SET_ERROR,
      payload: error,
    }),
    
    toggleSidebar: () => dispatch({
      type: actionTypes.TOGGLE_SIDEBAR,
    }),
    
    setSidebarOpen: (open) => dispatch({
      type: actionTypes.SET_SIDEBAR_OPEN,
      payload: open,
    }),
    
    // System Actions
    setSystemInfo: (info) => dispatch({
      type: actionTypes.SET_SYSTEM_INFO,
      payload: info,
    }),
    
    setModelStatus: (status) => dispatch({
      type: actionTypes.SET_MODEL_STATUS,
      payload: status,
    }),
    
    // Project Actions
    setCurrentProject: (project) => dispatch({
      type: actionTypes.SET_CURRENT_PROJECT,
      payload: project,
    }),
    
    createProject: (projectData) => dispatch({
      type: actionTypes.CREATE_PROJECT,
      payload: projectData,
    }),
    
    updateProject: (updates) => dispatch({
      type: actionTypes.UPDATE_PROJECT,
      payload: updates,
    }),
    
    // Processing Actions
    addToQueue: (item) => {
      const queueItem = {
        id: Date.now().toString(),
        timestamp: new Date().toISOString(),
        status: 'queued',
        progress: 0,
        ...item,
      };
      dispatch({
        type: actionTypes.ADD_TO_QUEUE,
        payload: queueItem,
      });
      return queueItem.id;
    },
    
    removeFromQueue: (id) => dispatch({
      type: actionTypes.REMOVE_FROM_QUEUE,
      payload: id,
    }),
    
    updateQueueItem: (id, updates) => dispatch({
      type: actionTypes.UPDATE_QUEUE_ITEM,
      payload: { id, updates },
    }),
    
    addToHistory: (item) => dispatch({
      type: actionTypes.ADD_TO_HISTORY,
      payload: {
        ...item,
        timestamp: new Date().toISOString(),
      },
    }),
    
    clearHistory: () => dispatch({
      type: actionTypes.CLEAR_HISTORY,
    }),
    
    // Upload Actions
    setUploadProgress: (progress) => dispatch({
      type: actionTypes.SET_UPLOAD_PROGRESS,
      payload: progress,
    }),
    
    addUploadedFile: (file) => {
      const uploadedFile = {
        id: Date.now().toString(),
        timestamp: new Date().toISOString(),
        ...file,
      };
      dispatch({
        type: actionTypes.ADD_UPLOADED_FILE,
        payload: uploadedFile,
      });
      return uploadedFile.id;
    },
    
    removeUploadedFile: (id) => dispatch({
      type: actionTypes.REMOVE_UPLOADED_FILE,
      payload: id,
    }),
    
    clearUploadedFiles: () => dispatch({
      type: actionTypes.CLEAR_UPLOADED_FILES,
    }),
    
    // Settings Actions
    updatePreferences: (preferences) => dispatch({
      type: actionTypes.UPDATE_PREFERENCES,
      payload: preferences,
    }),
    
    updateProcessingOptions: (options) => dispatch({
      type: actionTypes.UPDATE_PROCESSING_OPTIONS,
      payload: options,
    }),
    
    resetProcessingOptions: () => dispatch({
      type: actionTypes.RESET_PROCESSING_OPTIONS,
    }),
  };

  // Computed values
  const computed = {
    hasUploadedFiles: state.uploadedFiles.length > 0,
    isProcessingQueueEmpty: state.processingQueue.length === 0,
    hasActiveProcessing: state.processingQueue.some(
      item => item.status === 'processing'
    ),
    totalQueuedItems: state.processingQueue.length,
    completedProcessingCount: state.processingHistory.length,
    currentProjectHasFiles: state.currentProject?.files?.length > 0,
  };

  const value = {
    // State
    ...state,
    
    // Actions
    ...actions,
    
    // Computed values
    ...computed,
  };

  return (
    <AppContext.Provider value={value}>
      {children}
    </AppContext.Provider>
  );
};

export default AppContext;
