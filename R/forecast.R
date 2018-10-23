library('forecast')
library('tseries')

# Function for Root Mean Squared Error
rmse <- function(real, predicted) {
  sqrt(mean((real - predicted)^2))
}

# Function for Mean Absolute Error
mae <- function(real, predicted) {
  mean(abs(real - predicted))
}

# Load models
load("models/speed_arima_6.rda")
load("models/direction_arima_6.rda")

# Load testing data
test <- read.csv("../data/test_2018.csv", header=TRUE)
test$timestamp <- as.POSIXct(test$timestamp, format='%Y-%m-%d %H:%M')

### EVALUATION ###
tsdisplay(residuals(speed_model), lag.max=45)
tsdisplay(residuals(direc_model), lag.max=45)

# 
hist(residuals(speed_model))
qqnorm(residuals(speed_model));qqline(residuals(speed_model), col = 2)
shapiro.test(residuals(speed_model))

# Unit root
plot(speed_model)

# Forecast 1 day 
speed_forecast <- forecast(speed_model, h = 144)
plot(speed_forecast)
sp_pred <- as.data.frame(speed_forecast)
sp_pred$timestamp <- test$timestamp

direc_forecast <- forecast(direc_model, h = 144)
plot(direc_forecast)
di_pred <- as.data.frame(direc_forecast)
di_pred$timestamp <- test$timestamp

# Metrics
sprintf("SPEED:")
sprintf("Training:")
sprintf("RMSE: %f", rmse(speed_model$x, speed_model$fitted))
sprintf("MAE: %f", mae(speed_model$x, speed_model$fitted))
sprintf("Testing:")
sprintf("RMSE: %f", rmse(test$speed, speed_forecast$mean))
sprintf("MAE: %f", mae(test$speed, speed_forecast$mean))

sprintf("DIRECTION:")
sprintf("Training:")
sprintf("RMSE: %f", rmse(direc_model$x, direc_model$fitted))
sprintf("MAE: %f", mae(direc_model$x, direc_model$fitted))
sprintf("Testing:")
sprintf("RMSE: %f", rmse(test$direction, direc_forecast$mean))
sprintf("MAE: %f", mae(test$direction, direc_forecast$mean))

### FORECAST ###
# Speed
speed_forecast_plot <- ggplot() + 
  geom_line(data=sp_pred, aes(x=timestamp, y=`Point Forecast`), color = "blue") + 
  geom_line(data=test, aes(x=timestamp, y=speed), color = "red") + 
  xlab("Timestamp") + 
  ylab("Velocidad") 
speed_forecast_plot

# Direction
direc_forecast_plot <- ggplot() + 
  geom_line(data=di_pred, aes(x=timestamp, y=`Point Forecast`), color = "blue") + 
  geom_line(data=test, aes(x=timestamp, y=direction), color = "red") + 
  xlab("Timestamp") + 
  ylab("Direccion")
direc_forecast_plot

### SIMULATION ###
# Create simulated data for physical model 
for (sim in 1:100) { # 100 one day simulation
  sim_speed <- simulate(speed_model, nsim = 144, future = TRUE)
  sim_direc <- simulate(direc_model, nsim = 144, future = TRUE)
  write.csv(x=sim_speed, file=paste("../data/simulated/speed/", sim, ".csv", sep = ""))
  write.csv(x=sim_direc, file=paste("../data/simulated/direction/", sim, ".csv", sep = ""))
}