import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import axios from 'axios';
import {
  Typography, Box, Paper, Grid, FormControl, InputLabel,
  Select, MenuItem, Button, CircularProgress, Alert,
  Card, CardContent, CardHeader, Divider, LinearProgress,
  Tooltip, Chip, Stack, Tabs, Tab, Skeleton
} from '@mui/material';
import {
  Map as MapIcon,
  Speed as SpeedIcon,
  Straighten as StraightenIcon,
  Balance as BalanceIcon,
  TrackChanges as TrackChangesIcon,
  CompareArrows as CompareIcon,
  Refresh as RefreshIcon,
  ViewQuilt as GridIcon
} from '@mui/icons-material';
import { PieChart, Pie, Cell, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer } from 'recharts';

const TrackAnalysis = () => {
  const [loading, setLoading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [races, setRaces] = useState([]);
  const [selectedRace, setSelectedRace] = useState('');
  const [selectedYear, setSelectedYear] = useState(new Date().getFullYear());
  const [trackData, setTrackData] = useState(null);
  const [error, setError] = useState(null);
  const [trackVisualizations, setTrackVisualizations] = useState(null);
  const [loadingVisualizations, setLoadingVisualizations] = useState(false);
  const [selectedVisualization, setSelectedVisualization] = useState(0); // 0: Layout, 1: Characteristics, 2: Comparison
  const location = useLocation();

  // Use current year by default, but don't make it selectable
  useEffect(() => {
    // Get races for the current year
    fetchRaces(selectedYear);
    
    // Get race from URL query params if provided
    const queryParams = new URLSearchParams(location.search);
    const raceParam = queryParams.get('race');
    if (raceParam) {
      setSelectedRace(raceParam);
    }
  }, [location.search]);

  // When a race is selected, fetch track data automatically
  useEffect(() => {
    if (selectedRace) {
      fetchTrackData();
      fetchTrackVisualizations();
    }
  }, [selectedRace]);

  const fetchRaces = async (year) => {
    try {
      setLoading(true);
      const response = await axios.get(`/api/races?year=${year}`);
      setRaces(response.data.races);
      setLoading(false);
    } catch (err) {
      setError('Failed to fetch races: ' + (err.response?.data?.error || err.message));
      setLoading(false);
    }
  };

  const fetchTrackData = async () => {
    if (!selectedRace) return;
    
    try {
      setError(null);
      setAnalyzing(true); // Show loading state while fetching
      const response = await axios.get(`/api/track-data?race=${encodeURIComponent(selectedRace)}`);
      setTrackData(response.data);
      setAnalyzing(false);
    } catch (err) {
      if (err.response?.status === 404) {
        // Track data not available in database, automatically analyze
        console.log('Track data not available, running analysis...');
        await analyzeTrack();
      } else {
        setError('Failed to fetch track data: ' + (err.response?.data?.error || err.message));
        setAnalyzing(false);
      }
    }
  };

  const fetchTrackVisualizations = async () => {
    if (!selectedRace) return;
    
    try {
      setLoadingVisualizations(true);
      const response = await axios.get(`/api/all-track-visualizations?race=${encodeURIComponent(selectedRace)}&year=${selectedYear}`);
      setTrackVisualizations(response.data);
      setLoadingVisualizations(false);
    } catch (err) {
      console.error('Failed to fetch track visualizations:', err);
      setTrackVisualizations(null);
      setLoadingVisualizations(false);
    }
  };

  // Modified to be an internal function called by fetchTrackData when needed
  const analyzeTrack = async () => {
    if (!selectedRace) return;

    try {
      setAnalyzing(true);
      setError(null);
      
      const response = await axios.post('/api/analyze-track', {
        race: selectedRace,
        year: selectedYear - 1 // Use previous year for analysis
      });
      
      setTrackData(response.data.track_data);
      setAnalyzing(false);
    } catch (err) {
      setError('Track analysis failed: ' + (err.response?.data?.error || err.message));
      setAnalyzing(false);
    }
  };

  // Helper function to convert track characteristic to text description
  const getCharacteristicText = (value, characteristic) => {
    if (value === undefined || value === null) return 'Unknown';
    
    // Define thresholds for each characteristic
    const thresholds = {
      overtaking_difficulty: {
        low: 0.4, high: 0.7,
        labels: ['Easy', 'Moderate', 'Difficult']
      },
      tire_degradation: {
        low: 0.4, high: 0.7,
        labels: ['Low', 'Moderate', 'High']
      },
      qualifying_importance: {
        low: 0.6, high: 0.8,
        labels: ['Low', 'Moderate', 'High']
      },
      start_importance: {
        low: 0.6, high: 0.8,
        labels: ['Low', 'Moderate', 'High']
      },
      dirty_air_impact: {
        low: 0.5, high: 0.7,
        labels: ['Low', 'Moderate', 'High']
      }
    };
    
    const threshold = thresholds[characteristic] || { low: 0.4, high: 0.7, labels: ['Low', 'Moderate', 'High'] };
    
    if (value < threshold.low) return threshold.labels[0];
    if (value > threshold.high) return threshold.labels[2];
    return threshold.labels[1];
  };

  // Helper function to get chip color based on value
  const getChipColor = (value, isHigherBetter = false) => {
    if (value === undefined || value === null) return 'default';
    
    // For characteristics where lower is better (like overtaking difficulty)
    if (!isHigherBetter) {
      if (value < 0.4) return 'success';
      if (value > 0.7) return 'error';
      return 'warning';
    }
    
    // For characteristics where higher is better
    if (value > 0.7) return 'success';
    if (value < 0.4) return 'error';
    return 'warning';
  };

  // Handle visualization tab change
  const handleTabChange = (event, newValue) => {
    setSelectedVisualization(newValue);
  };

  // Get the appropriate visualization based on selected tab
  const getVisualization = () => {
    if (!trackVisualizations || !trackVisualizations.visualizations) {
      return null;
    }
    
    const visualizations = trackVisualizations.visualizations;
    
    switch (selectedVisualization) {
      case 0: 
        return visualizations.track;
      case 1:
        return visualizations.characteristics;
      case 2:
        return visualizations.comparison;
      default:
        return visualizations.track;
    }
  };

  // Prepare radar chart data if track data is available
  const radarData = trackData ? [
    {
      subject: 'Qualifying Importance',
      value: trackData.qualifying_importance || 0.5,
      fullMark: 1
    },
    {
      subject: 'Overtaking Difficulty',
      value: trackData.overtaking_difficulty || 0.5,
      fullMark: 1
    },
    {
      subject: 'Tire Degradation',
      value: trackData.tire_degradation || 0.5,
      fullMark: 1
    },
    {
      subject: 'Start Importance',
      value: trackData.start_importance || 0.5,
      fullMark: 1
    },
    {
      subject: 'Dirty Air Impact',
      value: trackData.dirty_air_impact || 0.5,
      fullMark: 1
    }
  ] : [];

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom>
        F1 Track Analysis
      </Typography>
      
      <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
        <Typography variant="h6" gutterBottom>
          Select Track
        </Typography>
        
        <Grid container spacing={2}>
          <Grid item xs={12}>
            <FormControl fullWidth disabled={loading}>
              <InputLabel>Race</InputLabel>
              <Select
                value={selectedRace}
                label="Race"
                onChange={(e) => setSelectedRace(e.target.value)}
              >
                {loading ? (
                  <MenuItem disabled>Loading races...</MenuItem>
                ) : (
                  races.map(race => (
                    <MenuItem key={race.round} value={race.name}>
                      {race.name} (Round {race.round})
                    </MenuItem>
                  ))
                )}
              </Select>
            </FormControl>
          </Grid>
        </Grid>
        
        {analyzing && (
          <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
            <CircularProgress size={24} /> 
            <Typography variant="body2" color="text.secondary" sx={{ ml: 1 }}>
              Loading track data...
            </Typography>
          </Box>
        )}
        
        {error && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {error}
          </Alert>
        )}
      </Paper>
      
      {/* Track Visualizations Section */}
      {selectedRace && (
        <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
          <Typography variant="h5" gutterBottom>
            Track Visualizations
          </Typography>
          
          <Tabs 
            value={selectedVisualization} 
            onChange={handleTabChange} 
            centered
            sx={{ mb: 3 }}
          >
            <Tab 
              icon={<TrackChangesIcon />} 
              label="Track Layout"
              disabled={!trackVisualizations || !trackVisualizations.visualizations?.track}
            />
            <Tab 
              icon={<SpeedIcon />} 
              label="Track Characteristics"
              disabled={!trackVisualizations || !trackVisualizations.visualizations?.characteristics}
            />
            <Tab 
              icon={<CompareIcon />} 
              label="Driver Comparison"
              disabled={!trackVisualizations || !trackVisualizations.visualizations?.comparison}
            />
          </Tabs>
          
          {loadingVisualizations ? (
            <Box sx={{ width: '100%', textAlign: 'center', py: 3 }}>
              <CircularProgress />
              <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
                Loading track visualizations...
              </Typography>
            </Box>
          ) : trackVisualizations ? (
            <Box sx={{ textAlign: 'center' }}>
              {getVisualization() ? (
                <img 
                  src={`data:image/png;base64,${getVisualization()}`} 
                  alt={`Track ${selectedVisualization === 0 ? 'Layout' : selectedVisualization === 1 ? 'Characteristics' : 'Comparison'}`} 
                  style={{ maxWidth: '100%', height: 'auto', borderRadius: '8px', boxShadow: '0 4px 8px rgba(0,0,0,0.1)' }}
                />
              ) : (
                <Alert severity="info">
                  No {selectedVisualization === 0 ? 'track layout' : selectedVisualization === 1 ? 'track characteristics' : 'driver comparison'} data available
                </Alert>
              )}
            </Box>
          ) : (
            <Alert severity="info">
              Select a race to view track visualizations
            </Alert>
          )}
          
          {trackVisualizations && selectedVisualization === 2 && trackVisualizations.comparison_drivers && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle2">
                Comparing top qualifying drivers:
              </Typography>
              <Stack direction="row" spacing={1} sx={{ mt: 1 }} justifyContent="center">
                {trackVisualizations.comparison_drivers.map((driver, index) => (
                  <Chip 
                    key={index} 
                    label={driver} 
                    variant="outlined" 
                    color={["primary", "secondary", "success"][index % 3]}
                  />
                ))}
              </Stack>
            </Box>
          )}
          
          <Box sx={{ mt: 3, textAlign: 'right' }}>
            <Button 
              variant="outlined" 
              startIcon={<RefreshIcon />}
              onClick={fetchTrackVisualizations}
              disabled={loadingVisualizations}
            >
              Refresh Track Visualizations
            </Button>
          </Box>
        </Paper>
      )}
      
      {trackData && (
        <Box>
          <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
            <Box sx={{ mb: 3 }}>
              <Typography variant="h5" gutterBottom>
                {selectedRace} Track Analysis
              </Typography>
              <Chip 
                label={trackData.track_type ? trackData.track_type.toUpperCase() : 'Unknown'} 
                color="primary" 
                variant="outlined"
                icon={<MapIcon />}
              />
            </Box>
            
            <Divider sx={{ mb: 3 }} />
            
            <Grid container spacing={4}>
              <Grid item xs={12} md={6}>
                <Typography variant="h6" gutterBottom>
                  Track Characteristics
                </Typography>
                
                <Box sx={{ mb: 3 }}>
                  <Grid container spacing={2}>
                    <Grid item xs={12}>
                      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
                        <Typography variant="body2">
                          Overtaking Difficulty
                        </Typography>
                        <Chip 
                          size="small" 
                          label={getCharacteristicText(trackData.overtaking_difficulty, 'overtaking_difficulty')} 
                          color={getChipColor(trackData.overtaking_difficulty)}
                        />
                      </Box>
                      <Tooltip title={`${(trackData.overtaking_difficulty * 100).toFixed(0)}%`}>
                        <LinearProgress 
                          variant="determinate" 
                          value={trackData.overtaking_difficulty * 100 || 0} 
                          color={getChipColor(trackData.overtaking_difficulty)}
                          sx={{ height: 8, borderRadius: 2 }}
                        />
                      </Tooltip>
                    </Grid>
                    
                    <Grid item xs={12}>
                      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
                        <Typography variant="body2">
                          Tire Degradation
                        </Typography>
                        <Chip 
                          size="small" 
                          label={getCharacteristicText(trackData.tire_degradation, 'tire_degradation')} 
                          color={getChipColor(trackData.tire_degradation)}
                        />
                      </Box>
                      <Tooltip title={`${(trackData.tire_degradation * 100).toFixed(0)}%`}>
                        <LinearProgress 
                          variant="determinate" 
                          value={trackData.tire_degradation * 100 || 0} 
                          color={getChipColor(trackData.tire_degradation)}
                          sx={{ height: 8, borderRadius: 2 }}
                        />
                      </Tooltip>
                    </Grid>
                    
                    <Grid item xs={12}>
                      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
                        <Typography variant="body2">
                          Qualifying Importance
                        </Typography>
                        <Chip 
                          size="small" 
                          label={getCharacteristicText(trackData.qualifying_importance, 'qualifying_importance')} 
                          color={getChipColor(trackData.qualifying_importance, true)}
                        />
                      </Box>
                      <Tooltip title={`${(trackData.qualifying_importance * 100).toFixed(0)}%`}>
                        <LinearProgress 
                          variant="determinate" 
                          value={trackData.qualifying_importance * 100 || 0} 
                          color={getChipColor(trackData.qualifying_importance, true)}
                          sx={{ height: 8, borderRadius: 2 }}
                        />
                      </Tooltip>
                    </Grid>
                    
                    <Grid item xs={12}>
                      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
                        <Typography variant="body2">
                          Race Start Importance
                        </Typography>
                        <Chip 
                          size="small" 
                          label={getCharacteristicText(trackData.start_importance, 'start_importance')} 
                          color={getChipColor(trackData.start_importance, true)}
                        />
                      </Box>
                      <Tooltip title={`${(trackData.start_importance * 100).toFixed(0)}%`}>
                        <LinearProgress 
                          variant="determinate" 
                          value={trackData.start_importance * 100 || 0} 
                          color={getChipColor(trackData.start_importance, true)}
                          sx={{ height: 8, borderRadius: 2 }}
                        />
                      </Tooltip>
                    </Grid>
                    
                    <Grid item xs={12}>
                      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
                        <Typography variant="body2">
                          Dirty Air Impact
                        </Typography>
                        <Chip 
                          size="small" 
                          label={getCharacteristicText(trackData.dirty_air_impact, 'dirty_air_impact')} 
                          color={getChipColor(trackData.dirty_air_impact)}
                        />
                      </Box>
                      <Tooltip title={`${(trackData.dirty_air_impact * 100).toFixed(0)}%`}>
                        <LinearProgress 
                          variant="determinate" 
                          value={trackData.dirty_air_impact * 100 || 0} 
                          color={getChipColor(trackData.dirty_air_impact)}
                          sx={{ height: 8, borderRadius: 2 }}
                        />
                      </Tooltip>
                    </Grid>
                  </Grid>
                </Box>
                
                {trackData.track_characteristics && (
                  <Box>
                    <Typography variant="subtitle1" gutterBottom>
                      Additional Track Info
                    </Typography>
                    
                    <Stack direction="row" spacing={1} flexWrap="wrap" sx={{ mb: 2 }}>
                      {trackData.track_characteristics.track_length && (
                        <Chip
                          icon={<StraightenIcon />}
                          label={`${trackData.track_characteristics.track_length.toFixed(3)} km`}
                          variant="outlined"
                          size="small"
                          sx={{ mb: 1 }}
                        />
                      )}
                      {trackData.track_characteristics.corners && (
                        <Chip
                          icon={<TrackChangesIcon />}
                          label={`${trackData.track_characteristics.corners} corners`}
                          variant="outlined"
                          size="small"
                          sx={{ mb: 1 }}
                        />
                      )}
                      {trackData.track_characteristics.country && (
                        <Chip
                          label={trackData.track_characteristics.country}
                          variant="outlined"
                          size="small"
                          sx={{ mb: 1 }}
                        />
                      )}
                    </Stack>
                  </Box>
                )}
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Box sx={{ height: 300, width: '100%' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <RadarChart cx="50%" cy="50%" outerRadius="80%" data={radarData}>
                      <PolarGrid strokeDasharray="3 3" />
                      <PolarAngleAxis dataKey="subject" />
                      <PolarRadiusAxis angle={30} domain={[0, 1]} />
                      <Radar name="Track Characteristics" dataKey="value" stroke="#8884d8" fill="#8884d8" fillOpacity={0.6} />
                    </RadarChart>
                  </ResponsiveContainer>
                </Box>
              </Grid>
            </Grid>
          </Paper>
          
          {trackData.overtaking_metrics && (
            <Paper elevation={2} sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Overtaking Analysis
              </Typography>
              
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6} md={3}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="h4" color="primary" gutterBottom align="center">
                        {trackData.overtaking_metrics.total_overtakes || 0}
                      </Typography>
                      <Typography variant="body2" color="text.secondary" align="center">
                        Total Overtakes
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                
                <Grid item xs={12} sm={6} md={3}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="h4" color="primary" gutterBottom align="center">
                        {trackData.overtaking_metrics.position_changes || 0}
                      </Typography>
                      <Typography variant="body2" color="text.secondary" align="center">
                        Position Changes
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                
                <Grid item xs={12} sm={6} md={3}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="h4" color="primary" gutterBottom align="center">
                        {trackData.overtaking_metrics.pit_stops || 0}
                      </Typography>
                      <Typography variant="body2" color="text.secondary" align="center">
                        Pit Stops
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                
                <Grid item xs={12} sm={6} md={3}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="h4" color="primary" gutterBottom align="center">
                        {trackData.overtaking_metrics.drs_effectiveness 
                          ? (trackData.overtaking_metrics.drs_effectiveness * 100).toFixed(1) + '%' 
                          : 'N/A'}
                      </Typography>
                      <Typography variant="body2" color="text.secondary" align="center">
                        DRS Effectiveness
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
            </Paper>
          )}
        </Box>
      )}
    </Box>
  );
};

export default TrackAnalysis;