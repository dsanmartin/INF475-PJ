library('forecast')
library('tseries')
library(ggplot2)

# Read data
train <- read.csv("../data/train_2018.csv", header=TRUE)
train$timestamp <- as.POSIXct(train$timestamp, format='%Y-%m-%d %H:%M')

# ACF and PACF for speed
ggAcf(train$speed, lag.max = 300, main = "")
ggPacf(train$speed, lag.max = 150, main = "")

speed_ACF <- acf(diff(train$speed, lag = 1), lag.max = 200, plot = FALSE)
plot(speed_ACF, main = "Speed Autocorrelation")
speed_PACF <- pacf(train$speed, lag.max = 200, plot = FALSE)
plot(speed_PACF, main = "Speed Partial Autocorrelation")

# Speed augmented Dickey-Fuller test
adf.test(train$speed, alternative = "stationary")

# Direction timeserie plot
plot.ts(train$direction)

# ACF and PACF for direction
ggAcf(train$direction, lag.max = 300, main = "")
ggPacf(train$direction, lag.max = 150, main = "")

# Direction augmented Dickey-Fuller test
adf.test(train$direction, alternative = "stationary")

# Time serie analysis
# Speed
speed_ts <- ts(train$speed, frequency = 24*60/10)
speed_dec <- stl(speed_ts, s.window="periodic")
plot(speed_dec)

acf(ts(speed_dec$time.series[, "remainder"]), lag.max=300)

plot.ts(speed_dec$time.series[, "remainder"])

deseasonal_speed <- seasadj(speed_dec)
plot.ts(deseasonal_speed)

acf(ts(deseasonal_speed), lag.max = 300)

# Direction
direc_ts <- ts(data$direction, frequency = 24*60/10)
direc_dec <- stl(direc_ts, s.window="periodic")
plot(direc_dec)

# Timeseries
speed <- ts(data$speed, frequency = 24*60/10)
direction <- ts(data$direction, frequency = 24*60/10)

# speed_ts <- ts(data$speed, frequency = 24*60/10)
# speed_dec <- stl(speed_ts, s.window="periodic")
# deseasonal_speed <- seasadj(speed_dec)

# # Models
# speed_model <- auto.arima(speed, seasonal=TRUE)#, D=1)
# speed_model
# 
# direc_model <- auto.arima(direction, seasonal=TRUE)
# direc_model
# 
# # Save models
# save(speed_model, file="models/speed_arima_6.rda")
# save(direc_model, file="models/direction_arima_6.rda")
