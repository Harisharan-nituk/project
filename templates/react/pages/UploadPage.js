
import React, { useState, useCallback, useRef } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  Grid,
  LinearProgress,
  Chip,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Alert,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemSecondaryAction,
  Divider,
  Container,
  Paper,
  Stepper,
  Step,
  StepLabel,
  StepContent,
} from '@mui/material';
import {
  CloudUpload,
  VideoFile,
  Image,
  Delete,
  PlayArrow,
  Stop,
  Settings,
  Face,
  Checkroom,
  Landscape,
  AutoAwesome,
  CheckCircle,
  Error,
  Info,
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';

import { useAppContext } from '../context/AppContext';
import { apiService } from '../services/apiService';
import ProcessingOptionsDialog from '../components/ProcessingOptionsDialog';
import VideoPreview from '../components/VideoPreview';

const UploadPage = () => {
  const {
    uploadedFiles,
    addUploadedFile,
    removeUploadedFile,
    clearUploadedFiles,
    setUploadProgress,
    uploadProgress,
    processingOptions,
    updateProcessingOptions,
    addToQueue,
    createProject,
    setError,
    userPreferences,
  } = useAppContext();

  const [activeStep, setActiveStep] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [previewFile, setPreviewFile] = useState(null);
  const [optionsDialogOpen, setOptionsDialogOpen] = useState(false);
  const [projectName, setProjectName] = useState('');
  const [autoProcess, setAutoProcess] = useState(userPreferences.processOnUpload);

  const fileInputRef = useRef(null);

  const steps = [
    {
      label: 'Upload Videos',
      description: 'Select and upload video files for processing',
    },
    {
      label: 'Configure Options',
      description: 'Set processing options like face swap, clothing, background',
    },
    {
      label: 'Review & Process',
      description: 'Review settings and start processing',
    },
  ];

  const onDrop = useCallback(async (acceptedFiles) => {
    setIsUploading(true);
    setUploadProgress(0);

    for (let i = 0; i < acceptedFiles.length; i++) {
      const file = acceptedFiles[i];
      
      try {
        // Create form data
        const formData = new FormData();
        formData.append('video', file);

        // Upload file with progress tracking
        const response = await apiService.uploadFile(formData, (progressEvent) => {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadProgress(progress);
        });

        // Add to uploaded files
        const uploadedFile = {
          id: response.file_id,
          name: file.name,
          size: file.size,
          type: file.type,
          uploadedAt: new Date().toISOString(),
          status: 'uploaded',
          path: response.file_path,
          thumbnail: null, // Will be generated later
        };

        addUploadedFile(uploadedFile);

        // Generate thumbnail
        try {
          const thumbnailUrl = await apiService.generateThumbnail(response.file_id);
          // Update file with thumbnail
          // This would require an update function in context
        } catch (error) {
          console.error('Failed to generate thumbnail:', error);
        }

      } catch (error) {
        console.error('Upload failed:', error);
        setError(`Failed to upload ${file.name}: ${error.message}`);
      }
    }

    setIsUploading(false);
    setUploadProgress(0);
    
    // Auto-advance to next step if files were uploaded successfully
    if (acceptedFiles.length > 0 && activeStep === 0) {
      setActiveStep(1);
    }
  }, [addUploadedFile, setUploadProgress, setError, activeStep]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv'],
    },
    multiple: true,
    disabled: isUploading,
  });

  const handleRemoveFile = (fileId) => {
    removeUploadedFile(fileId);
    if (previewFile?.id === fileId) {
      setPreviewFile(null);
    }
  };

  const handlePreviewFile = (file) => {
    setPreviewFile(file);
  };

  const handleStartProcessing = async () => {
    if (uploadedFiles.length === 0) {
      setError('No files to process');
      return;
    }

    try {
      // Create project if name is provided
      let project = null;
      if (projectName.trim()) {
        project = createProject({
          name: projectName.trim(),
          files: uploadedFiles.map(f => f.id),
          processingOptions,
        });
      }

      // Add each file to processing queue
      for (const file of uploadedFiles) {
        const queueItem = {
          fileId: file.id,
          fileName: file.name,
          projectId: project?.id,
          options: processingOptions,
          type: getProcessingType(processingOptions),
        };

        const queueId = addToQueue(queueItem);

        // Start processing via API
        try {
          await apiService.startProcessing(file.id, processingOptions);
        } catch (error) {
          console.error('Failed to start processing:', error);
          setError(`Failed to start processing ${file.name}`);
        }
      }

      // Clear uploaded files and reset form
      clearUploadedFiles();
      setProjectName('');
      setActiveStep(0);
      
    } catch (error) {
      console.error('Failed to start processing:', error);
      setError('Failed to start processing');
    }
  };

  const getProcessingType = (options) => {
    if (options.faceImage) return 'face_swap';
    if (options.clothingStyle) return 'clothing_change';
    if (options.background) return 'background_change';
    return 'general';
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getProcessingOptionsCount = () => {
    let count = 0;
    if (processingOptions.faceImage) count++;
    if (processingOptions.clothingStyle) count++;
    if (processingOptions.background) count++;
    if (processingOptions.textPrompt) count++;
    return count;
  };

  const renderUploadStep = () => (
    <Card>
      <CardContent>
        <Box
          {...getRootProps()}
          sx={{
            border: '2px dashed',
            borderColor: isDragActive ? 'primary.main' : 'grey.600',
            borderRadius: 2,
            p: 4,
            textAlign: 'center',
            cursor: 'pointer',
            transition: 'all 0.3s ease',
            background: isDragActive 
              ? 'linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(236, 72, 153, 0.1) 100%)'
              : 'transparent',
            '&:hover': {
              borderColor: 'primary.main',
              background: 'rgba(99, 102, 241, 0.05)',
            },
          }}
        >
          <input {...getInputProps()} />
          <motion.div
            animate={isDragActive ? { scale: 1.05 } : { scale: 1 }}
            transition={{ type: 'spring', stiffness: 300 }}
          >
            <CloudUpload
              sx={{
                fontSize: 64,
                color: isDragActive ? 'primary.main' : 'text.secondary',
                mb: 2,
              }}
            />
            <Typography variant="h5" gutterBottom>
              {isDragActive ? 'Drop files here' : 'Upload Video Files'}
            </Typography>
            <Typography variant="body1" color="text.secondary" gutterBottom>
              Drag & drop video files here, or click to browse
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Supported formats: MP4, AVI, MOV, WMV, FLV, MKV
            </Typography>
          </motion.div>
        </Box>

        {isUploading && (
          <Box sx={{ mt: 3 }}>
            <Typography variant="body2" gutterBottom>
              Uploading... {uploadProgress}%
            </Typography>
            <LinearProgress
              variant="determinate"
              value={uploadProgress}
              sx={{
                height: 8,
                borderRadius: 4,
                '& .MuiLinearProgress-bar': {
                  background: 'linear-gradient(135deg, #6366f1 0%, #ec4899 100%)',
                },
              }}
            />
          </Box>
        )}

        {uploadedFiles.length > 0 && (
          <Box sx={{ mt: 3 }}>
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
              <Typography variant="h6">
                Uploaded Files ({uploadedFiles.length})
              </Typography>
              <Button
                size="small"
                color="error"
                onClick={clearUploadedFiles}
                startIcon={<Delete />}
              >
                Clear All
              </Button>
            </Box>

            <List>
              {uploadedFiles.map((file, index) => (
                <React.Fragment key={file.id}>
                  <ListItem>
                    <ListItemIcon>
                      <VideoFile color="primary" />
                    </ListItemIcon>
                    <ListItemText
                      primary={file.name}
                      secondary={
                        <Box>
                          <Typography variant="body2" color="text.secondary">
                            {formatFileSize(file.size)} • Uploaded {new Date(file.uploadedAt).toLocaleTimeString()}
                          </Typography>
                          <Chip
                            label={file.status}
                            color="success"
                            size="small"
                            sx={{ mt: 0.5 }}
                          />
                        </Box>
                      }
                    />
                    <ListItemSecondaryAction>
                      <IconButton
                        edge="end"
                        onClick={() => handlePreviewFile(file)}
                        sx={{ mr: 1 }}
                      >
                        <PlayArrow />
                      </IconButton>
                      <IconButton
                        edge="end"
                        color="error"
                        onClick={() => handleRemoveFile(file.id)}
                      >
                        <Delete />
                      </IconButton>
                    </ListItemSecondaryAction>
                  </ListItem>
                  {index < uploadedFiles.length - 1 && <Divider />}
                </React.Fragment>
              ))}
            </List>
          </Box>
        )}
      </CardContent>
    </Card>
  );

  const renderOptionsStep = () => (
    <Card>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
          <Typography variant="h5">Processing Options</Typography>
          <Button
            variant="outlined"
            startIcon={<Settings />}
            onClick={() => setOptionsDialogOpen(true)}
          >
            Configure
          </Button>
        </Box>

        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <Paper
              sx={{
                p: 2,
                textAlign: 'center',
                border: processingOptions.faceImage ? '2px solid' : '1px solid',
                borderColor: processingOptions.faceImage ? 'primary.main' : 'grey.600',
                background: processingOptions.faceImage 
                  ? 'rgba(99, 102, 241, 0.1)' 
                  : 'transparent',
              }}
            >
              <Face sx={{ fontSize: 48, mb: 1, color: 'primary.main' }} />
              <Typography variant="h6" gutterBottom>
                Face Swap
              </Typography>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Replace faces in videos with custom face images
              </Typography>
              {processingOptions.faceImage ? (
                <Chip label="Configured" color="primary" size="small" />
              ) : (
                <Chip label="Not Set" variant="outlined" size="small" />
              )}
            </Paper>
          </Grid>

          <Grid item xs={12} md={4}>
            <Paper
              sx={{
                p: 2,
                textAlign: 'center',
                border: processingOptions.clothingStyle ? '2px solid' : '1px solid',
                borderColor: processingOptions.clothingStyle ? 'secondary.main' : 'grey.600',
                background: processingOptions.clothingStyle 
                  ? 'rgba(236, 72, 153, 0.1)' 
                  : 'transparent',
              }}
            >
              <Checkroom sx={{ fontSize: 48, mb: 1, color: 'secondary.main' }} />
              <Typography variant="h6" gutterBottom>
                Clothing Change
              </Typography>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Change clothing styles and outfits in videos
              </Typography>
              {processingOptions.clothingStyle ? (
                <Chip label="Configured" color="secondary" size="small" />
              ) : (
                <Chip label="Not Set" variant="outlined" size="small" />
              )}
            </Paper>
          </Grid>

          <Grid item xs={12} md={4}>
            <Paper
              sx={{
                p: 2,
                textAlign: 'center',
                border: processingOptions.background ? '2px solid' : '1px solid',
                borderColor: processingOptions.background ? 'success.main' : 'grey.600',
                background: processingOptions.background 
                  ? 'rgba(34, 197, 94, 0.1)' 
                  : 'transparent',
              }}
            >
              <Landscape sx={{ fontSize: 48, mb: 1, color: 'success.main' }} />
              <Typography variant="h6" gutterBottom>
                Background Change
              </Typography>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Replace or modify video backgrounds
              </Typography>
              {processingOptions.background ? (
                <Chip label="Configured" color="success" size="small" />
              ) : (
                <Chip label="Not Set" variant="outlined" size="small" />
              )}
            </Paper>
          </Grid>

          <Grid item xs={12}>
            <Paper sx={{ p: 2 }}>
              <Box display="flex" alignItems="center" mb={2}>
                <AutoAwesome sx={{ mr: 1, color: 'warning.main' }} />
                <Typography variant="h6">AI Text Prompt</Typography>
              </Box>
              <TextField
                fullWidth
                multiline
                rows={3}
                placeholder="Describe the changes you want to make to the video..."
                value={processingOptions.textPrompt || ''}
                onChange={(e) => updateProcessingOptions({ textPrompt: e.target.value })}
                variant="outlined"
              />
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                Example: "Change to formal business attire in an office setting"
              </Typography>
            </Paper>
          </Grid>
        </Grid>

        {getProcessingOptionsCount() === 0 && (
          <Alert severity="info" sx={{ mt: 2 }}>
            <Typography variant="body2">
              No processing options configured. Click "Configure" to set up face swap, clothing change, or background replacement options.
            </Typography>
          </Alert>
        )}
      </CardContent>
    </Card>
  );

  const renderReviewStep = () => (
    <Card>
      <CardContent>
        <Typography variant="h5" gutterBottom>
          Review & Process
        </Typography>

        <Grid container spacing={3}>
          <Grid item xs={12} md={8}>
            <Paper sx={{ p: 2, mb: 2 }}>
              <Typography variant="h6" gutterBottom>
                Files to Process
              </Typography>
              <List dense>
                {uploadedFiles.map((file) => (
                  <ListItem key={file.id}>
                    <ListItemIcon>
                      <CheckCircle color="success" />
                    </ListItemIcon>
                    <ListItemText
                      primary={file.name}
                      secondary={formatFileSize(file.size)}
                    />
                  </ListItem>
                ))}
              </List>
            </Paper>

            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Processing Configuration
              </Typography>
              <Grid container spacing={2}>
                {processingOptions.faceImage && (
                  <Grid item xs={12} sm={6}>
                    <Box display="flex" alignItems="center">
                      <Face sx={{ mr: 1, color: 'primary.main' }} />
                      <Typography variant="body2">Face Swap: Enabled</Typography>
                    </Box>
                  </Grid>
                )}
                {processingOptions.clothingStyle && (
                  <Grid item xs={12} sm={6}>
                    <Box display="flex" alignItems="center">
                      <Checkroom sx={{ mr: 1, color: 'secondary.main' }} />
                      <Typography variant="body2">
                        Clothing: {processingOptions.clothingStyle}
                      </Typography>
                    </Box>
                  </Grid>
                )}
                {processingOptions.background && (
                  <Grid item xs={12} sm={6}>
                    <Box display="flex" alignItems="center">
                      <Landscape sx={{ mr: 1, color: 'success.main' }} />
                      <Typography variant="body2">Background: Custom</Typography>
                    </Box>
                  </Grid>
                )}
                {processingOptions.textPrompt && (
                  <Grid item xs={12}>
                    <Box display="flex" alignItems="flex-start">
                      <AutoAwesome sx={{ mr: 1, mt: 0.5, color: 'warning.main' }} />
                      <Box>
                        <Typography variant="body2" fontWeight="bold">
                          AI Prompt:
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          {processingOptions.textPrompt}
                        </Typography>
                      </Box>
                    </Box>
                  </Grid>
                )}
              </Grid>
              
              {getProcessingOptionsCount() === 0 && (
                <Alert severity="warning" sx={{ mt: 2 }}>
                  No processing options configured. Videos will be processed with default settings.
                </Alert>
              )}
            </Paper>
          </Grid>

          <Grid item xs={12} md={4}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Project Settings
              </Typography>
              
              <TextField
                fullWidth
                label="Project Name (Optional)"
                value={projectName}
                onChange={(e) => setProjectName(e.target.value)}
                placeholder="My Video Project"
                sx={{ mb: 2 }}
              />

              <FormControlLabel
                control={
                  <Switch
                    checked={autoProcess}
                    onChange={(e) => setAutoProcess(e.target.checked)}
                  />
                }
                label="Auto-process uploads"
                sx={{ mb: 2 }}
              />

              <Divider sx={{ my: 2 }} />

              <Typography variant="body2" color="text.secondary" gutterBottom>
                Estimated Processing Time
              </Typography>
              <Typography variant="h6" color="primary.main" gutterBottom>
                ~{uploadedFiles.length * 2} minutes
              </Typography>

              <Typography variant="body2" color="text.secondary" gutterBottom>
                Total Files: {uploadedFiles.length}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Options: {getProcessingOptionsCount()}
              </Typography>
            </Paper>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );

  return (
    <Container maxWidth="lg">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        {/* Header */}
        <Box sx={{ mb: 4 }}>
          <Typography variant="h1" gutterBottom>
            Upload & Process
          </Typography>
          <Typography variant="h6" color="text.secondary">
            Upload videos and configure AI processing options
          </Typography>
        </Box>

        {/* Stepper */}
        <Paper sx={{ p: 3, mb: 4 }}>
          <Stepper activeStep={activeStep} orientation="horizontal">
            {steps.map((step, index) => (
              <Step key={step.label}>
                <StepLabel
                  onClick={() => setActiveStep(index)}
                  sx={{ cursor: 'pointer' }}
                >
                  {step.label}
                </StepLabel>
              </Step>
            ))}
          </Stepper>
        </Paper>

        {/* Step Content */}
        <AnimatePresence mode="wait">
          <motion.div
            key={activeStep}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.3 }}
          >
            {activeStep === 0 && renderUploadStep()}
            {activeStep === 1 && renderOptionsStep()}
            {activeStep === 2 && renderReviewStep()}
          </motion.div>
        </AnimatePresence>

        {/* Navigation */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 4 }}>
          <Button
            disabled={activeStep === 0}
            onClick={() => setActiveStep(activeStep - 1)}
            variant="outlined"
          >
            Back
          </Button>

          <Box>
            {activeStep < steps.length - 1 ? (
              <Button
                variant="contained"
                onClick={() => setActiveStep(activeStep + 1)}
                disabled={activeStep === 0 && uploadedFiles.length === 0}
                sx={{ ml: 1 }}
              >
                Next
              </Button>
            ) : (
              <Button
                variant="contained"
                onClick={handleStartProcessing}
                disabled={uploadedFiles.length === 0}
                startIcon={<PlayArrow />}
                sx={{
                  ml: 1,
                  background: 'linear-gradient(135deg, #6366f1 0%, #ec4899 100%)',
                  '&:hover': {
                    background: 'linear-gradient(135deg, #4f46e5 0%, #db2777 100%)',
                  },
                }}
              >
                Start Processing
              </Button>
            )}
          </Box>
        </Box>

        {/* Processing Options Dialog */}
        <ProcessingOptionsDialog
          open={optionsDialogOpen}
          onClose={() => setOptionsDialogOpen(false)}
          options={processingOptions}
          onOptionsChange={updateProcessingOptions}
        />

        {/* Video Preview Dialog */}
        <Dialog
          open={!!previewFile}
          onClose={() => setPreviewFile(null)}
          maxWidth="md"
          fullWidth
        >
          <DialogTitle>
            <Box display="flex" justifyContent="space-between" alignItems="center">
              <Typography variant="h6">Video Preview</Typography>
              <IconButton onClick={() => setPreviewFile(null)}>
                <Stop />
              </IconButton>
            </Box>
          </DialogTitle>
          <DialogContent>
            {previewFile && (
              <VideoPreview
                file={previewFile}
                onClose={() => setPreviewFile(null)}
              />
            )}
          </DialogContent>
        </Dialog>

        {/* Status Messages */}
        {uploadedFiles.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            <Alert
              severity="success"
              sx={{
                mt: 2,
                background: 'rgba(34, 197, 94, 0.1)',
                border: '1px solid rgba(34, 197, 94, 0.2)',
              }}
            >
              <Typography variant="body2">
                {uploadedFiles.length} file(s) uploaded successfully and ready for processing.
              </Typography>
            </Alert>
          </motion.div>
        )}

        {isUploading && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <Alert
              severity="info"
              sx={{
                mt: 2,
                background: 'rgba(99, 102, 241, 0.1)',
                border: '1px solid rgba(99, 102, 241, 0.2)',
              }}
            >
              <Typography variant="body2">
                Upload in progress... Please wait.
              </Typography>
            </Alert>
          </motion.div>
        )}
      </motion.div>
    </Container>
  );
};

export default UploadPage;
