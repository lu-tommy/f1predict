import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import axios from 'axios';
import { 
  Typography, Grid, Card, CardContent, CardActions, 
  Button, Box, CircularProgress, Paper, Chip
} from '@mui/material';
import SportsMotoIcon from '@mui/icons-material/SportsMotorsports';
import MapIcon from '@mui/icons-material/Map';
import TimelineIcon from '@mui/icons-material/Timeline';
import HistoryIcon from '@mui/icons-material/History';
import FutureIcon from '@mui/icons-material/TimerOutlined';

const Dashboard = () => {
  const [loading, setLoading] = useState(true);
  const [races, setRaces] = useState([]);
  const [error, setError] = useState(null);
  const currentYear = new Date().getFullYear();

  useEffect(() => {
    const fetchRaces = async () => {
      try {
        setLoading(true);
        const response = await axios.get(`/api/races?year=${currentYear}`);
        setRaces(response.data.races);
        setError(null);
      } catch (err) {
        setError('Failed to fetch race schedule. ' + (err.response?.data?.error || err.message));
      } finally {
        setLoading(false);
      }
    };

    fetchRaces();
  }, [currentYear]);

  return (
    <Box>
      <Box sx={{ mb: 4, textAlign: 'center' }}>
        <Typography variant="h4" component="h1" gutterBottom>
          F1 Race Prediction Dashboard
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Predict race results and analyze track characteristics with machine learning.
        </Typography>
      </Box>

      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <CardContent sx={{ flexGrow: 1 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <TimelineIcon color="primary" fontSize="large" sx={{ mr: 1 }} />
                <Typography variant="h5" component="h2">
                  Race Prediction
                </Typography>
              </Box>
              <Typography variant="body2">
                Predict race positions using machine learning and historical data. Get insights on potential winners and team performance.
              </Typography>
            </CardContent>
            <CardActions>
              <Button component={Link} to="/predict" size="small" color="primary">
                Make a Prediction
              </Button>
            </CardActions>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <CardContent sx={{ flexGrow: 1 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <MapIcon color="primary" fontSize="large" sx={{ mr: 1 }} />
                <Typography variant="h5" component="h2">
                  Track Analysis
                </Typography>
              </Box>
              <Typography variant="body2">
                Analyze track characteristics including overtaking difficulty, tire degradation, and qualifying importance.
              </Typography>
            </CardContent>
            <CardActions>
              <Button component={Link} to="/track-analysis" size="small" color="primary">
                Analyze Tracks
              </Button>
            </CardActions>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <CardContent sx={{ flexGrow: 1 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <SportsMotoIcon color="primary" fontSize="large" sx={{ mr: 1 }} />
                <Typography variant="h5" component="h2">
                  Driver & Team Data
                </Typography>
              </Box>
              <Typography variant="body2">
                Access current driver and team information dynamically loaded from FastF1. No hardcoded data ensures accuracy.
              </Typography>
            </CardContent>
            <CardActions>
              <Button component={Link} to="/predict" size="small" color="primary">
                View Drivers
              </Button>
            </CardActions>
          </Card>
        </Grid>
      </Grid>

      <Paper elevation={2} sx={{ p: 3 }}>
        <Typography variant="h5" component="h2" gutterBottom>
          F1 {currentYear} Season Calendar
        </Typography>
        
        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
            <CircularProgress />
          </Box>
        ) : error ? (
          <Typography color="error">{error}</Typography>
        ) : (
          <Grid container spacing={2} sx={{ mt: 2 }}>
            {races.map((race) => (
              <Grid item xs={12} sm={6} md={4} key={race.round}>
                <Card variant="outlined" sx={{ height: '100%' }}>
                  <CardContent>
                    <Typography variant="h6" component="div">
                      {race.name}
                    </Typography>
                    <Typography color="text.secondary" gutterBottom>
                      Round {race.round} • {race.country}
                    </Typography>
                    {race.date && (
                      <Typography variant="body2">
                        {new Date(race.date).toLocaleDateString('en-US', { 
                          weekday: 'long', 
                          year: 'numeric', 
                          month: 'long', 
                          day: 'numeric' 
                        })}
                      </Typography>
                    )}
                    <Box sx={{ mt: 2 }}>
                      <Chip 
                        size="small" 
                        label={race.isPast ? 'Completed' : 'Upcoming'} 
                        color={race.isPast ? 'default' : 'primary'} 
                        icon={race.isPast ? <HistoryIcon fontSize="small" /> : <FutureIcon fontSize="small" />}
                      />
                    </Box>
                  </CardContent>
                  <CardActions>
                    <Button 
                      size="small" 
                      component={Link} 
                      to={`/predict?race=${encodeURIComponent(race.name)}`}
                    >
                      Predict
                    </Button>
                    <Button 
                      size="small" 
                      component={Link} 
                      to={`/track-analysis?race=${encodeURIComponent(race.name)}`}
                    >
                      Track Data
                    </Button>
                  </CardActions>
                </Card>
              </Grid>
            ))}
          </Grid>
        )}
      </Paper>
    </Box>
  );
};

export default Dashboard;