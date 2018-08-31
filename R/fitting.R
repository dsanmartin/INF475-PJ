library('forecast')
library('tseries')

# Read data
data <- read.csv("../data/2017_30min.csv", header=TRUE)
#colnames(data) <- c("timestamp", "speed", "direction")
data$timestamp <- as.POSIXct(data$timestamp)

# Timeseries
speed <- ts(data$speed, frequency = 24*60/30)
direction <- ts(data$direction, frequency = 24*60/30)

# Models
speed_model <- auto.arima(speed)
speed_model

direc_model <- auto.arima(direction)
direc_model

# Save models
save(speed_model, file="models/speed_arima.rda")
save(direc_model, file="models/direction_arima.rda")