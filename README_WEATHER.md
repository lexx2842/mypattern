# Weather Data Integration

## OpenWeatherMap API Setup

To get real weather data for Zurich, you need to:

1. **Register for a free API key:**
   - Go to https://openweathermap.org/api
   - Sign up for a free account
   - Get your API key from the dashboard

2. **Set the environment variable:**
   ```bash
   export OPENWEATHER_API_KEY=your_api_key_here
   ```

3. **Or create a .env file:**
   ```
   OPENWEATHER_API_KEY=your_api_key_here
   ```

## Features

- **Real weather data** for Zurich (temperature, humidity, air quality)
- **Fallback to sample data** if no API key is provided
- **Error handling** for API failures
- **Air pollution data** from OpenWeatherMap

## Data Sources

- **Weather**: OpenWeatherMap Current Weather API
- **Air Quality**: OpenWeatherMap Air Pollution API
- **Pollen**: Currently using sample data (no free pollen API available)

## Free Tier Limits

- 1,000 API calls per day
- 60 calls per minute
- Sufficient for daily weather sync


