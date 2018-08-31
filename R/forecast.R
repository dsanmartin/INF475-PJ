library('forecast')
library('tseries')

# Load models
load("models/speed_arima.rda")
load("models/direction_arima.rda")

# Evaluate model
#adf.test(direc_model$x, alternative = "stationary")
tsdisplay(residuals(speed_model), lag.max=45)
tsdisplay(residuals(direc_model), lag.max=45)

# Forecast
speed_forecast <- forecast(speed_model, h=96)
plot(speed_forecast)

direc_forecast <- forecast(direc_model, h=96)
plot(direc_forecast)

# Simulation
sim_speed <- simulate(speed_model, future = FALSE)
sim_direc <- simulate(direc_model, future = FALSE)

# Save simulated
write.csv(x=sim_speed, file="../data/simulated/speed.csv")
write.csv(x=sim_direc, file="../data/simulated/direction.csv")

plot(sim_speed, col = 'red')
plot(sim_direc, col = 'green')