import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import axios from 'axios';
import { 
  Typography, Box, Paper, Grid, FormControl, InputLabel, 
  Select, MenuItem, Button, CircularProgress, Alert,
  Table, TableBody, TableCell, TableContainer, TableHead, 
  TableRow, Chip, Card, CardContent, Divider, Tabs, Tab,
  LinearProgress, Badge, IconButton, Tooltip, Switch, FormControlLabel
} from '@mui/material';
import { 
  EmojiEvents as TrophyIcon,
  SportsScore as RaceIcon,
  Speed as SpeedIcon,
  History as HistoryIcon,
  TimerOutlined as FutureIcon,
  CompareArrows as CompareIcon,
  CheckCircle as CorrectIcon,
  Cancel as WrongIcon,
  SportsScore as PodiumIcon,
  Assessment as AccuracyIcon
} from '@mui/icons-material';

const RacePrediction = () => {
  const [loading, setLoading] = useState(false);
  const [loadingDrivers, setLoadingDrivers] = useState(false);
  const [predicting, setPredicting] = useState(false);
  const [races, setRaces] = useState([]);
  const [drivers, setDrivers] = useState([]);
  const [selectedRace, setSelectedRace] = useState('');
  const [selectedYear, setSelectedYear] = useState(new Date().getFullYear());
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);
  const [showComparison, setShowComparison] = useState(false);
  const location = useLocation();

  useEffect(() => {
    // Get races for the selected year
    fetchRaces(selectedYear);
    
    // Get race from URL query params if provided
    const queryParams = new URLSearchParams(location.search);
    const raceParam = queryParams.get('race');
    if (raceParam) {
      setSelectedRace(raceParam);
    }
  }, [selectedYear, location.search]);

  useEffect(() => {
    if (selectedRace) {
      // Clear old race data immediately
      setDrivers([]);
      setPrediction(null);
      setError(null);
  
      fetchDrivers();
      handlePredict();
    }
  }, [selectedRace, showComparison]); // Add showComparison to dependencies
  
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

  const fetchDrivers = async () => {
    if (!selectedRace) return;
    
    try {
      setLoadingDrivers(true);
      const response = await axios.get(`/api/drivers?race=${encodeURIComponent(selectedRace)}&year=${selectedYear}`);
      setDrivers(response.data.drivers);
      setLoadingDrivers(false);
    } catch (err) {
      setError('Failed to fetch driver data: ' + (err.response?.data?.error || err.message));
      setLoadingDrivers(false);
    }
  };

  const handlePredict = async () => {
    if (!selectedRace) {
      setError('Please select a race first');
      return;
    }

    try {
      setPredicting(true);
      setError(null);
      
      // Add compare flag for past races
      const isPastRace = isRacePast(selectedRace);
      const shouldCompare = isPastRace && showComparison;
      
      const response = await axios.post('/api/predict', {
        race: selectedRace,
        year: selectedYear,
        compare: shouldCompare // Send flag to request comparison
      });
      
      setPrediction(response.data);
      setPredicting(false);
    } catch (err) {
      setError('Prediction failed: ' + (err.response?.data?.error || err.message));
      setPredicting(false);
    }
  };
  
  // Check if a race is in the past or future - re-enable this function
  const isRacePast = (raceName) => {
    if (!races || !races.length) return false;
    const race = races.find(r => r.name === raceName);
    return race ? race.isPast : false;
  };

  // Function to get team color based on team name
  const getTeamColor = (teamName) => {
    if (!teamName) return '#999999';
    
    const teamLower = teamName.toLowerCase();
    if (teamLower.includes('mercedes')) return '#00D2BE';
    if (teamLower.includes('red bull')) return '#0600EF';
    if (teamLower.includes('ferrari')) return '#DC0000';
    if (teamLower.includes('mclaren')) return '#FF8700';
    if (teamLower.includes('aston')) return '#006F62';
    if (teamLower.includes('alpine')) return '#0090FF';
    if (teamLower.includes('williams')) return '#005AFF';
    if (teamLower.includes('alpha') || teamLower.includes('alphatauri')) return '#2B4562';
    if (teamLower.includes('alfa')) return '#900000';
    if (teamLower.includes('haas')) return '#FFFFFF';
    return '#999999'; // Default
  };

  // Find team for a driver
  const getDriverTeam = (driverCode) => {
    if (!drivers || !drivers.length) return null;
    const driver = drivers.find(d => d.code === driverCode);
    return driver ? driver.team : null;
  };

  // Generate years for dropdown
  const years = [];
  const currentYear = new Date().getFullYear();
  for (let year = currentYear; year >= currentYear - 2; year--) {
    years.push(year);
  }

  // Format accuracy metric as percentage
  const formatAccuracy = (value) => {
    return `${(value * 100).toFixed(0)}%`;
  };

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom>
        F1 Race Prediction
      </Typography>
      
      <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
        <Typography variant="h6" gutterBottom>
          Select Race
        </Typography>
        
        <Grid container spacing={2}>
          <Grid item xs={12} md={4}>
            <FormControl fullWidth>
              <InputLabel>Year</InputLabel>
              <Select
                value={selectedYear}
                label="Year"
                onChange={(e) => setSelectedYear(e.target.value)}
              >
                {years.map(year => (
                  <MenuItem key={year} value={year}>{year}</MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item xs={12} md={8}>
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
                      {race.isPast && " ✓"}
                    </MenuItem>
                  ))
                )}
              </Select>
            </FormControl>
          </Grid>
          
          {isRacePast(selectedRace) && (
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={showComparison}
                    onChange={() => setShowComparison(!showComparison)}
                    color="primary"
                  />
                }
                label={
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <CompareIcon sx={{ mr: 1 }} />
                    <Typography variant="body2">
                      Compare predictions with actual results
                    </Typography>
                  </Box>
                }
              />
            </Grid>
          )}
        </Grid>
        
        {error && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {error}
          </Alert>
        )}
        
        {predicting && (
          <Box sx={{ display: 'flex', alignItems: 'center', mt: 2 }}>
            <CircularProgress size={24} sx={{ mr: 2 }} />
            <Typography variant="body2" color="text.secondary">
              {showComparison ? "Comparing predictions with actual results..." : "Loading data..."}
            </Typography>
          </Box>
        )}
      </Paper>
      
      {loadingDrivers && (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
          <CircularProgress />
        </Box>
      )}
      
      {drivers.length > 0 && !loadingDrivers && (
        <Paper elevation={2} sx={{ p: 3, mb: 4 }}>
          <Typography variant="h6" gutterBottom>
            Current Drivers for {selectedRace}
          </Typography>
          
          <Grid container spacing={2}>
            {drivers.map(driver => (
              <Grid item xs={6} sm={4} md={3} key={driver.code}>
                <Card 
                  variant="outlined" 
                  sx={{ 
                    height: '100%',
                    borderLeft: `4px solid ${getTeamColor(driver.team)}`
                  }}
                >
                  <CardContent>
                    <Typography variant="h6" component="div">
                      {driver.code}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      {driver.team}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Paper>
      )}
      
      {/* Comparison View */}
      {prediction && prediction.comparison && (
        <Box>
          <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
            <Typography variant="h5" gutterBottom>
              Prediction Accuracy: {selectedRace} {selectedYear}
            </Typography>
            
            <Box sx={{ mb: 4 }}>
              <Grid container spacing={3}>
                <Grid item xs={12} sm={6} md={3}>
                  <Card>
                    <CardContent>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                        <Badge 
                          badgeContent={
                            prediction.accuracy.winner_correct ? 
                            <CorrectIcon color="success" fontSize="small" /> : 
                            <WrongIcon color="error" fontSize="small" />
                          }
                          anchorOrigin={{
                            vertical: 'top',
                            horizontal: 'right',
                          }}
                          sx={{ width: '100%' }}
                        >
                          <Typography variant="h6" color="primary">
                            Winner
                          </Typography>
                        </Badge>
                      </Box>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <Box>
                          <Typography variant="body2" color="text.secondary">
                            Predicted
                          </Typography>
                          <Typography variant="body1" fontWeight="bold">
                            {prediction.prediction.winner}
                          </Typography>
                        </Box>
                        <Box>
                          <Typography variant="body2" color="text.secondary">
                            Actual
                          </Typography>
                          <Typography variant="body1" fontWeight="bold">
                            {prediction.actual.winner}
                          </Typography>
                        </Box>
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
                
                <Grid item xs={12} sm={6} md={3}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6" color="primary" gutterBottom>
                        Podium Accuracy
                      </Typography>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                        <LinearProgress 
                          variant="determinate" 
                          value={prediction.accuracy.podium_accuracy * 100}
                          color={prediction.accuracy.podium_accuracy > 0.66 ? "success" : 
                                prediction.accuracy.podium_accuracy > 0.33 ? "warning" : "error"}
                          sx={{ height: 10, borderRadius: 5, width: '100%', mr: 1 }}
                        />
                        <Typography variant="body1" sx={{ minWidth: 45 }}>
                          {formatAccuracy(prediction.accuracy.podium_accuracy)}
                        </Typography>
                      </Box>
                      <Typography variant="body2" color="text.secondary">
                        {(prediction.accuracy.podium_accuracy * 3).toFixed(0)} of 3 podium positions correct
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                
                <Grid item xs={12} sm={6} md={3}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6" color="primary" gutterBottom>
                        Top 10 Accuracy
                      </Typography>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                        <LinearProgress 
                          variant="determinate" 
                          value={prediction.accuracy.top_10_accuracy * 100}
                          color={prediction.accuracy.top_10_accuracy > 0.7 ? "success" : 
                                prediction.accuracy.top_10_accuracy > 0.5 ? "warning" : "error"}
                          sx={{ height: 10, borderRadius: 5, width: '100%', mr: 1 }}
                        />
                        <Typography variant="body1" sx={{ minWidth: 45 }}>
                          {formatAccuracy(prediction.accuracy.top_10_accuracy)}
                        </Typography>
                      </Box>
                      <Typography variant="body2" color="text.secondary">
                        {(prediction.accuracy.top_10_accuracy * 10).toFixed(0)} of 10 points positions correct
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                
                <Grid item xs={12} sm={6} md={3}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6" color="primary" gutterBottom>
                        Position Error
                      </Typography>
                      <Typography variant="h4" align="center">
                        {prediction.accuracy.position_avg_error.toFixed(1)}
                      </Typography>
                      <Typography variant="body2" color="text.secondary" align="center">
                        Average position error
                      </Typography>
                      <Typography variant="body2" sx={{ mt: 1 }}>
                        {prediction.accuracy.matched_positions} exact position matches
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
            </Box>
            
            {prediction.comparisonVisualization && (
              <Box sx={{ mt: 3, mb: 4, textAlign: 'center' }}>
                <Typography variant="h6" gutterBottom>
                  Prediction vs Actual Results
                </Typography>
                <img 
                  src={`data:image/png;base64,${prediction.comparisonVisualization}`} 
                  alt="Prediction vs Results Comparison" 
                  style={{ maxWidth: '100%', height: 'auto' }}
                />
              </Box>
            )}
            
            <Divider sx={{ mb: 3 }} />
            
            <Box>
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <Typography variant="h6" gutterBottom>
                    Predicted Results
                  </Typography>
                  <TableContainer>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>Pos</TableCell>
                          <TableCell>Driver</TableCell>
                          <TableCell>Team</TableCell>
                          <TableCell align="right">Score</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {prediction.prediction.positions.slice(0, 10).map((pos) => {
                          // Find actual position for this driver
                          const actualPos = prediction.actual.positions.find(p => p.driver === pos.driver);
                          const positionMatch = actualPos && actualPos.position === pos.position;
                          
                          return (
                            <TableRow 
                              key={pos.driver}
                              sx={{
                                bgcolor: pos.position <= 3 ? 'rgba(255, 215, 0, 0.05)' : 'inherit',
                                '&:hover': {
                                  bgcolor: 'rgba(0, 0, 0, 0.04)',
                                },
                              }}
                            >
                              <TableCell>
                                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                  {pos.position <= 3 ? (
                                    <Chip 
                                      size="small" 
                                      label={pos.position} 
                                      color={pos.position === 1 ? 'warning' : 'default'}
                                      sx={{ 
                                        fontWeight: 'bold',
                                        bgcolor: pos.position === 1 ? 'gold' : 
                                                pos.position === 2 ? 'silver' : 
                                                pos.position === 3 ? '#cd7f32' : 'default',
                                        color: 'white',
                                        mr: 1
                                      }}
                                    />
                                  ) : (
                                    <Typography sx={{ mr: 1 }}>{pos.position}</Typography>
                                  )}
                                  
                                  {positionMatch && (
                                    <Tooltip title="Exact position match">
                                      <CorrectIcon color="success" fontSize="small" />
                                    </Tooltip>
                                  )}
                                </Box>
                              </TableCell>
                              <TableCell>
                                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                  <Box 
                                    sx={{ 
                                      width: 4, 
                                      height: 16, 
                                      bgcolor: getTeamColor(getDriverTeam(pos.driver)),
                                      mr: 1,
                                      borderRadius: 1
                                    }} 
                                  />
                                  {pos.driver}
                                </Box>
                              </TableCell>
                              <TableCell>{getDriverTeam(pos.driver)}</TableCell>
                              <TableCell align="right">{pos.score.toFixed(3)}</TableCell>
                            </TableRow>
                          );
                        })}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </Grid>
                
                <Grid item xs={12} md={6}>
                  <Typography variant="h6" gutterBottom>
                    Actual Results
                  </Typography>
                  <TableContainer>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>Pos</TableCell>
                          <TableCell>Driver</TableCell>
                          <TableCell>Team</TableCell>
                          <TableCell align="right">Points</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {prediction.actual.positions.slice(0, 10).map((pos) => {
                          // Find predicted position for this driver
                          const predPos = prediction.prediction.positions.find(p => p.driver === pos.driver);
                          const positionMatch = predPos && predPos.position === pos.position;
                          
                          return (
                            <TableRow 
                              key={pos.driver}
                              sx={{
                                bgcolor: pos.position <= 3 ? 'rgba(255, 215, 0, 0.05)' : 'inherit',
                                '&:hover': {
                                  bgcolor: 'rgba(0, 0, 0, 0.04)',
                                },
                              }}
                            >
                              <TableCell>
                                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                  {pos.position <= 3 ? (
                                    <Chip 
                                      size="small" 
                                      label={pos.position} 
                                      color={pos.position === 1 ? 'warning' : 'default'}
                                      sx={{ 
                                        fontWeight: 'bold',
                                        bgcolor: pos.position === 1 ? 'gold' : 
                                                pos.position === 2 ? 'silver' : 
                                                pos.position === 3 ? '#cd7f32' : 'default',
                                        color: 'white',
                                        mr: 1
                                      }}
                                    />
                                  ) : (
                                    <Typography sx={{ mr: 1 }}>{pos.position}</Typography>
                                  )}
                                  
                                  {positionMatch && (
                                    <Tooltip title="Exact position match">
                                      <CorrectIcon color="success" fontSize="small" />
                                    </Tooltip>
                                  )}
                                </Box>
                              </TableCell>
                              <TableCell>
                                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                  <Box 
                                    sx={{ 
                                      width: 4, 
                                      height: 16, 
                                      bgcolor: getTeamColor(pos.team),
                                      mr: 1,
                                      borderRadius: 1
                                    }} 
                                  />
                                  {pos.driver}
                                </Box>
                              </TableCell>
                              <TableCell>{pos.team}</TableCell>
                              <TableCell align="right">
                                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end' }}>
                                  {pos.points}
                                  {pos.fastestLap && (
                                    <Tooltip title="Fastest Lap">
                                      <SpeedIcon 
                                        color="secondary" 
                                        fontSize="small" 
                                        sx={{ ml: 1 }} 
                                      />
                                    </Tooltip>
                                  )}
                                </Box>
                              </TableCell>
                            </TableRow>
                          );
                        })}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </Grid>
              </Grid>
            </Box>
          </Paper>
        </Box>
      )}
      
      {/* Regular Prediction or Results View (non-comparison) */}
      {prediction && !prediction.comparison && (
        <Box>
          <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
            {/* Banner showing whether this is actual results or prediction */}
            <Box sx={{ mb: 2 }}>
              <Chip
                icon={prediction.isPrediction ? <FutureIcon /> : <HistoryIcon />}
                label={prediction.isPrediction ? "PREDICTION" : "ACTUAL RESULTS"}
                color={prediction.isPrediction ? "primary" : "secondary"}
                variant="outlined"
                sx={{ fontWeight: 'bold', fontSize: '0.9rem', mb: 2 }}
              />
            </Box>
            
            <Box 
              sx={{ 
                display: 'flex', 
                alignItems: 'center', 
                mb: 2
              }}
            >
              <TrophyIcon color="warning" sx={{ mr: 1, fontSize: 30 }} />
              <Typography variant="h5">
                {prediction.isPrediction ? 'Predicted' : 'Race'} Winner: {prediction.winner}
              </Typography>
            </Box>
            
            <Divider sx={{ mb: 3 }} />
            
            <Typography variant="h6" gutterBottom>
              {prediction.isPrediction ? 'Predicted Results' : 'Official Race Results'}
            </Typography>
            
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Position</TableCell>
                    <TableCell>Driver</TableCell>
                    <TableCell>Team</TableCell>
                    {prediction.isPrediction ? (
                      <TableCell align="right">Prediction Score</TableCell>
                    ) : (
                      <TableCell align="right">Points</TableCell>
                    )}
                    {!prediction.isPrediction && (
                      <TableCell align="center">Fastest Lap</TableCell>
                    )}
                  </TableRow>
                </TableHead>
                <TableBody>
                  {prediction.positions.map((pos) => (
                    <TableRow 
                      key={pos.driver}
                      sx={{
                        bgcolor: pos.position <= 3 ? 'rgba(255, 215, 0, 0.05)' : 'inherit',
                        '&:hover': {
                          bgcolor: 'rgba(0, 0, 0, 0.04)',
                        },
                      }}
                    >
                      <TableCell>
                        {pos.position <= 3 ? (
                          <Chip 
                            size="small" 
                            label={pos.position} 
                            color={pos.position === 1 ? 'warning' : 'default'}
                            sx={{ 
                              fontWeight: 'bold',
                              bgcolor: pos.position === 1 ? 'gold' : 
                                      pos.position === 2 ? 'silver' : 
                                      pos.position === 3 ? '#cd7f32' : 'default',
                              color: 'white'
                            }}
                          />
                        ) : (
                          pos.position
                        )}
                      </TableCell>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <Box 
                            sx={{ 
                              width: 4, 
                              height: 16, 
                              bgcolor: getTeamColor(pos.team || getDriverTeam(pos.driver)),
                              mr: 1,
                              borderRadius: 1
                            }} 
                          />
                          {pos.driver}
                        </Box>
                      </TableCell>
                      <TableCell>{pos.team || getDriverTeam(pos.driver)}</TableCell>
                      {prediction.isPrediction ? (
                        <TableCell align="right">{pos.score.toFixed(3)}</TableCell>
                      ) : (
                        <TableCell align="right">{pos.points}</TableCell>
                      )}
                      {!prediction.isPrediction && (
                        <TableCell align="center">
                          {pos.fastestLap && (
                            <Chip 
                              size="small" 
                              label="FL" 
                              color="secondary"
                              sx={{ fontSize: '0.7rem' }}
                            />
                          )}
                        </TableCell>
                      )}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
          
          {prediction.visualization && (
            <Paper elevation={2} sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                {prediction.isPrediction ? 'Prediction Visualization' : 'Race Results Visualization'}
              </Typography>
              <Box sx={{ textAlign: 'center', mt: 2 }}>
                <img 
                  src={`data:image/png;base64,${prediction.visualization}`} 
                  alt={prediction.isPrediction ? "Race prediction visualization" : "Race results visualization"} 
                  style={{ maxWidth: '100%', height: 'auto' }}
                />
              </Box>
            </Paper>
          )}
        </Box>
      )}
    </Box>
  );
};

export default RacePrediction;