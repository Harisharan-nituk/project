import React, { useState } from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Badge,
  Menu,
  MenuItem,
  Avatar,
  Box,
  Chip,
  Tooltip,
  Button,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Notifications,
  AccountCircle,
  Settings,
  Logout,
  DarkMode,
  LightMode,
  Speed,
  Memory,
  CloudQueue,
} from '@mui/icons-material';
import { motion } from 'framer-motion';

import { useAppContext } from '../context/AppContext';

const Navbar = ({ onMenuClick }) => {
  const {
    systemInfo,
    hasActiveProcessing,
    totalQueuedItems,
    userPreferences,
    updatePreferences,
  } = useAppContext();

  const [anchorEl, setAnchorEl] = useState(null);
  const [notificationAnchor, setNotificationAnchor] = useState(null);

  const handleProfileMenuOpen = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const handleNotificationClick = (event) => {
    setNotificationAnchor(event.currentTarget);
  };

  const handleNotificationClose = () => {
    setNotificationAnchor(null);
  };

  const toggleTheme = () => {
    updatePreferences({
      theme: userPreferences.theme === 'dark' ? 'light' : 'dark'
    });
  };

  const getSystemStatusColor = () => {
    if (!systemInfo) return 'error';
    if (systemInfo.video_generator_ready && systemInfo.cuda_available) return 'success';
    if (systemInfo.video_generator_ready) return 'warning';
    return 'error';
  };

  const getSystemStatusText = () => {
    if (!systemInfo) return 'Offline';
    if (systemInfo.video_generator_ready && systemInfo.cuda_available) return 'GPU Ready';
    if (systemInfo.video_generator_ready) return 'CPU Ready';
    return 'Offline';
  };

  return (
    <AppBar
      position="fixed"
      elevation={0}
      sx={{
        zIndex: (theme) => theme.zIndex.drawer + 1,
        background: 'rgba(15, 23, 42, 0.9)',
        backdropFilter: 'blur(20px)',
        borderBottom: '1px solid rgba(148, 163, 184, 0.1)',
      }}
    >
      <Toolbar>
        {/* Menu Button */}
        <IconButton
          color="inherit"
          aria-label="open drawer"
          onClick={onMenuClick}
          edge="start"
          sx={{ mr: 2 }}
        >
          <MenuIcon />
        </IconButton>

        {/* Logo and Title */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Box display="flex" alignItems="center" sx={{ flexGrow: 1 }}>
            <Box
              sx={{
                width: 32,
                height: 32,
                borderRadius: '8px',
                background: 'linear-gradient(135deg, #6366f1 0%, #ec4899 100%)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                mr: 2,
              }}
            >
              <Typography
                variant="h6"
                sx={{
                  fontWeight: 'bold',
                  color: 'white',
                  fontSize: '1.2rem',
                }}
              >
                V
              </Typography>
            </Box>
            <Typography
              variant="h6"
              noWrap
              sx={{
                fontWeight: 700,
                background: 'linear-gradient(135deg, #6366f1 0%, #ec4899 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}
            >
              Video Generation Studio
            </Typography>
          </Box>
        </motion.div>

        {/* System Status */}
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <Tooltip title="System Status">
            <Chip
              icon={<Speed />}
              label={getSystemStatusText()}
              color={getSystemStatusColor()}
              variant="outlined"
              size="small"
              sx={{ mr: 2 }}
            />
          </Tooltip>
        </motion.div>

        {/* Processing Queue Status */}
        {totalQueuedItems > 0 && (
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3 }}
          >
            <Tooltip title="Items in processing queue">
              <Chip
                icon={<CloudQueue />}
                label={`${totalQueuedItems} queued`}
                color={hasActiveProcessing ? 'warning' : 'info'}
                variant="filled"
                size="small"
                sx={{ mr: 2 }}
              />
            </Tooltip>
          </motion.div>
        )}

        {/* GPU Status */}
        {systemInfo?.cuda_available && (
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3, delay: 0.1 }}
          >
            <Tooltip title={`GPU: ${systemInfo.gpu_name || 'Available'}`}>
              <Chip
                icon={<Memory />}
                label="GPU"
                color="success"
                variant="outlined"
                size="small"
                sx={{ mr: 2 }}
              />
            </Tooltip>
          </motion.div>
        )}

        {/* Theme Toggle */}
        <Tooltip title="Toggle theme">
          <IconButton color="inherit" onClick={toggleTheme} sx={{ mr: 1 }}>
            {userPreferences.theme === 'dark' ? <LightMode /> : <DarkMode />}
          </IconButton>
        </Tooltip>

        {/* Notifications */}
        <Tooltip title="Notifications">
          <IconButton
            color="inherit"
            onClick={handleNotificationClick}
            sx={{ mr: 1 }}
          >
            <Badge badgeContent={hasActiveProcessing ? 1 : 0} color="secondary">
              <Notifications />
            </Badge>
          </IconButton>
        </Tooltip>

        {/* Profile Menu */}
        <Tooltip title="Account settings">
          <IconButton
            size="large"
            edge="end"
            aria-label="account of current user"
            aria-controls="profile-menu"
            aria-haspopup="true"
            onClick={handleProfileMenuOpen}
            color="inherit"
          >
            <Avatar
              sx={{
                width: 32,
                height: 32,
                background: 'linear-gradient(135deg, #6366f1 0%, #ec4899 100%)',
              }}
            >
              <AccountCircle />
            </Avatar>
          </IconButton>
        </Tooltip>
      </Toolbar>

      {/* Profile Menu */}
      <Menu
        id="profile-menu"
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleMenuClose}
        transformOrigin={{ horizontal: 'right', vertical: 'top' }}
        anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
        PaperProps={{
          sx: {
            mt: 1,
            background: 'rgba(30, 41, 59, 0.95)',
            backdropFilter: 'blur(20px)',
            border: '1px solid rgba(148, 163, 184, 0.1)',
            borderRadius: 2,
            minWidth: 200,
          },
        }}
      >
        <MenuItem onClick={handleMenuClose}>
          <Box display="flex" alignItems="center" width="100%">
            <Avatar
              sx={{
                width: 24,
                height: 24,
                mr: 2,
                background: 'linear-gradient(135deg, #6366f1 0%, #ec4899 100%)',
              }}
            >
              <AccountCircle />
            </Avatar>
            <Box>
              <Typography variant="body2" fontWeight="bold">
                User
              </Typography>
              <Typography variant="caption" color="text.secondary">
                user@example.com
              </Typography>
            </Box>
          </Box>
        </MenuItem>
        
        <MenuItem onClick={handleMenuClose}>
          <Settings sx={{ mr: 2 }} />
          Settings
        </MenuItem>
        
        <MenuItem onClick={handleMenuClose}>
          <Logout sx={{ mr: 2 }} />
          Logout
        </MenuItem>
      </Menu>

      {/* Notifications Menu */}
      <Menu
        id="notifications-menu"
        anchorEl={notificationAnchor}
        open={Boolean(notificationAnchor)}
        onClose={handleNotificationClose}
        transformOrigin={{ horizontal: 'right', vertical: 'top' }}
        anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
        PaperProps={{
          sx: {
            mt: 1,
            background: 'rgba(30, 41, 59, 0.95)',
            backdropFilter: 'blur(20px)',
            border: '1px solid rgba(148, 163, 184, 0.1)',
            borderRadius: 2,
            minWidth: 300,
            maxHeight: 400,
          },
        }}
      >
        {hasActiveProcessing ? (
          <MenuItem onClick={handleNotificationClose}>
            <Box>
              <Typography variant="body2" fontWeight="bold">
                Processing in Progress
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {totalQueuedItems} item(s) in queue
              </Typography>
            </Box>
          </MenuItem>
        ) : (
          <MenuItem onClick={handleNotificationClose}>
            <Box>
              <Typography variant="body2" color="text.secondary">
                No new notifications
              </Typography>
            </Box>
          </MenuItem>
        )}
      </Menu>
    </AppBar>
  );
};

export default Navbar;