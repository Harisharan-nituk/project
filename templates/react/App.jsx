import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline, Box, CircularProgress, Alert, Snackbar } from '@mui/material';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

// Components
import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';
import HomePage from './pages/HomePage';
import UploadPage from './pages/UploadPage';
import ProcessingPage from './pages/ProcessingPage';
import GalleryPage from './pages/GalleryPage';
import SettingsPage from './pages/SettingsPage';
import LoadingOverlay from './components/LoadingOverlay';

// Services
import { apiService } from './services/apiService';
import { socketService } from './services/socketService';

// Context
import { AppProvider, useAppContext } from './context/AppContext';

// Theme configuration
const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#6366f1',
      light: '#818cf8',
      dark: '#4f46e5',
    },
    secondary: {
      main: '#ec4899',
      light: '#f472b6',
      dark: '#db2777',
    },
    background: {
      default: '#0f172a',
      paper: '#1e293b',
    },
    text: {
      primary: '#f8fafc',
      secondary: '#cbd5e1',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 700,
      background: 'linear-gradient(135deg, #6366f1 0%, #ec4899 100%)',
      WebkitBackgroundClip: 'text',
      WebkitTextFillColor: 'transparent',
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 600,
    },
    h3: {
      fontSize: '1.5rem',
      fontWeight: 600,
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: '12px',
          textTransform: 'none',
          fontWeight: 600,
          padding: '12px 24px',
          background: 'linear-gradient(135deg, #6366f1 0%, #ec4899 100%)',
          '&:hover': {
            background: 'linear-gradient(135deg, #4f46e5 0%, #db2777 100%)',
            transform: 'translateY(-2px)',
            boxShadow: '0 10px 20px rgba(99, 102, 241, 0.3)',
          },
          transition: 'all 0.3s ease',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: '16px',
          background: 'rgba(30, 41, 59, 0.8)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(148, 163, 184, 0.1)',
          transition: 'all 0.3s ease',
          '&:hover': {
            transform: 'translateY(-4px)',
            boxShadow: '0 20px 40px rgba(0, 0, 0, 0.3)',
            border: '1px solid rgba(99, 102, 241, 0.3)',
          },
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          background: 'rgba(15, 23, 42, 0.9)',
          backdropFilter: 'blur(20px)',
          borderBottom: '1px solid rgba(148, 163, 184, 0.1)',
        },
      },
    },
  },
});

function AppContent() {
  const {
    isLoading,
    error,
    setError,
    systemInfo,
    setSystemInfo,
    sidebarOpen,
    setSidebarOpen
  } = useAppContext();

  const [notification, setNotification] = useState({
    open: false,
    message: '',
    severity: 'info'
  });

  useEffect(() => {
    // Initialize system info
    const loadSystemInfo = async () => {
      try {
        const info = await apiService.getSystemInfo();
        setSystemInfo(info);
      } catch (err) {
        console.error('Failed to load system info:', err);
        setError('Failed to load system information');
      }
    };

    loadSystemInfo();

    // Initialize socket connection
    socketService.connect();

    // Socket event listeners
    socketService.on('processing_update', (data) => {
      setNotification({
        open: true,
        message: `Processing update: ${data.message}`,
        severity: 'info'
      });
    });

    socketService.on('processing_complete', (data) => {
      setNotification({
        open: true,
        message: 'Video processing completed successfully!',
        severity: 'success'
      });
    });

    socketService.on('processing_error', (data) => {
      setNotification({
        open: true,
        message: `Processing error: ${data.error}`,
        severity: 'error'
      });
    });

    // Cleanup
    return () => {
      socketService.disconnect();
    };
  }, [setSystemInfo, setError]);

  const handleCloseNotification = () => {
    setNotification({ ...notification, open: false });
  };

  const handleCloseError = () => {
    setError(null);
  };

  if (isLoading) {
    return <LoadingOverlay message="Initializing application..." />;
  }

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      <Navbar onMenuClick={() => setSidebarOpen(!sidebarOpen)} />
      
      <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />
      
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          mt: 8,
          ml: sidebarOpen ? { sm: '280px' } : 0,
          transition: 'margin 0.3s ease',
          background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)',
          minHeight: 'calc(100vh - 64px)',
        }}
      >
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/upload" element={<UploadPage />} />
          <Route path="/processing" element={<ProcessingPage />} />
          <Route path="/gallery" element={<GalleryPage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Box>

      {/* Notifications */}
      <Snackbar
        open={notification.open}
        autoHideDuration={6000}
        onClose={handleCloseNotification}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert
          onClose={handleCloseNotification}
          severity={notification.severity}
          variant="filled"
          sx={{ borderRadius: '12px' }}
        >
          {notification.message}
        </Alert>
      </Snackbar>

      {/* Error notification */}
      <Snackbar
        open={!!error}
        autoHideDuration={8000}
        onClose={handleCloseError}
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
      >
        <Alert
          onClose={handleCloseError}
          severity="error"
          variant="filled"
          sx={{ borderRadius: '12px' }}
        >
          {error}
        </Alert>
      </Snackbar>

      <ToastContainer
        position="bottom-right"
        autoClose={5000}
        hideProgressBar={false}
        newestOnTop
        closeOnClick
        rtl={false}
        pauseOnFocusLoss
        draggable
        pauseOnHover
        theme="dark"
      />
    </Box>
  );
}

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <AppProvider>
        <Router>
          <AppContent />
        </Router>
      </AppProvider>
    </ThemeProvider>
  );
}

export default App;