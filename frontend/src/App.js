import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import axios from 'axios';
import { 
  AppBar, Toolbar, Typography, Container, Box, Drawer, List, 
  ListItem, ListItemIcon, ListItemText, CssBaseline, CircularProgress,
  Alert, AlertTitle, IconButton, Divider, useTheme
} from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import DashboardIcon from '@mui/icons-material/Dashboard';
import SportsMotoIcon from '@mui/icons-material/SportsMotorsports';
import MapIcon from '@mui/icons-material/Map';
import TimelineIcon from '@mui/icons-material/Timeline';
import Dashboard from './components/Dashboard';
import RacePrediction from './components/RacePrediction';
import TrackAnalysis from './components/TrackAnalysis';
import './App.css';

// API base URL
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';
axios.defaults.baseURL = API_URL;

function App() {
  const theme = useTheme();
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [backendStatus, setBackendStatus] = useState({
    checked: false,
    online: false,
    fastf1Available: false,
    predictionModulesAvailable: false,
    error: null
  });

  useEffect(() => {
    // Check backend status on component mount
    checkBackendStatus();
  }, []);

  const checkBackendStatus = async () => {
    try {
      const response = await axios.get('/api/status');
      setBackendStatus({
        checked: true,
        online: response.data.status === 'online',
        fastf1Available: response.data.fastf1_available,
        predictionModulesAvailable: response.data.prediction_modules_available,
        error: null
      });
    } catch (error) {
      setBackendStatus({
        checked: true,
        online: false,
        fastf1Available: false,
        predictionModulesAvailable: false,
        error: error.message
      });
    }
  };

  const toggleDrawer = () => {
    setDrawerOpen(!drawerOpen);
  };

  const drawerContent = (
    <Box sx={{ width: 250 }} role="presentation">
      <Box sx={{ 
        p: 2, 
        bgcolor: theme.palette.primary.main, 
        color: 'white',
        display: 'flex',
        alignItems: 'center'
      }}>
        <SportsMotoIcon sx={{ mr: 1 }} />
        <Typography variant="h6" component="div">
          F1 Predictor
        </Typography>
      </Box>
      <Divider />
      <List>
        <ListItem button component={Link} to="/" onClick={toggleDrawer}>
          <ListItemIcon>
            <DashboardIcon />
          </ListItemIcon>
          <ListItemText primary="Dashboard" />
        </ListItem>
        <ListItem button component={Link} to="/predict" onClick={toggleDrawer}>
          <ListItemIcon>
            <TimelineIcon />
          </ListItemIcon>
          <ListItemText primary="Race Prediction" />
        </ListItem>
        <ListItem button component={Link} to="/track-analysis" onClick={toggleDrawer}>
          <ListItemIcon>
            <MapIcon />
          </ListItemIcon>
          <ListItemText primary="Track Analysis" />
        </ListItem>
      </List>
    </Box>
  );

  return (
    <Router>
      <CssBaseline />
      <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
        <AppBar position="static">
          <Toolbar>
            <IconButton
              size="large"
              edge="start"
              color="inherit"
              aria-label="menu"
              sx={{ mr: 2 }}
              onClick={toggleDrawer}
            >
              <MenuIcon />
            </IconButton>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              F1 Race Prediction System
            </Typography>
          </Toolbar>
        </AppBar>
        
        <Drawer anchor="left" open={drawerOpen} onClose={toggleDrawer}>
          {drawerContent}
        </Drawer>
        
        <Container component="main" sx={{ flexGrow: 1, py: 3 }}>
          {!backendStatus.checked ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
              <CircularProgress />
            </Box>
          ) : !backendStatus.online ? (
            <Alert severity="error">
              <AlertTitle>Connection Error</AlertTitle>
              Cannot connect to the backend server. Please make sure the Flask server is running.
              {backendStatus.error && <Box mt={1}>Error: {backendStatus.error}</Box>}
            </Alert>
          ) : !backendStatus.fastf1Available ? (
            <Alert severity="warning">
              <AlertTitle>FastF1 Not Available</AlertTitle>
              The FastF1 package is not available on the backend server. Some functionality may be limited.
            </Alert>
          ) : !backendStatus.predictionModulesAvailable ? (
            <Alert severity="warning">
              <AlertTitle>Prediction Modules Not Available</AlertTitle>
              The prediction modules are not available on the backend server. Prediction functionality will be limited.
            </Alert>
          ) : (
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/predict" element={<RacePrediction />} />
              <Route path="/track-analysis" element={<TrackAnalysis />} />
            </Routes>
          )}
        </Container>
        
        <Box component="footer" sx={{ py: 3, px: 2, mt: 'auto', backgroundColor: theme.palette.grey[200] }}>
          <Container maxWidth="sm">
            <Typography variant="body2" color="text.secondary" align="center">
              F1 Race Prediction System © {new Date().getFullYear()}
            </Typography>
          </Container>
        </Box>
      </Box>
    </Router>
  );
}

export default App;