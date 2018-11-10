library('forecast')
library('tseries')
library('ggplot2')

# Function for Root Mean Squared Error
rmse <- function(real, predicted) {
  sqrt(mean((real - predicted)^2))
}

# Function for Mean Absolute Error
mae <- function(real, predicted) {
  mean(abs(real - predicted))
}

# For Q-Q plot
qqplot.data <- function (vec) { # argument: vector of numbers
  # following four lines from base R's qqline()
  y <- quantile(vec[!is.na(vec)], c(0.25, 0.75))
  x <- qnorm(c(0.25, 0.75))
  slope <- diff(y)/diff(x)
  int <- y[1L] - slope * x[1L]
  
  d <- data.frame(resids = vec)
  
  ggplot(d, aes(sample = resids)) + 
    stat_qq() + 
    geom_abline(slope = slope, intercept = int) +
    xlab("Theoretical Quantiles") +
    ylab("Sample Quantiles")
  
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

# Residual Histogram Speed
res_speed = as.data.frame(residuals(speed_model))
ggplot(res_speed, aes(x = x)) + 
  geom_histogram(aes(y = ..density..),
  #qplot(res_speed$x,
  #      geom = "histogram",
                 #bins=20,
                 fill = I("blue"),
                 col = I("blue"),
                 alpha=I(.5)
                 ) + 
  xlab("Speed Model Residuals") +
  ylab("Frecuency") + 
  stat_function(fun = dnorm, args = list(mean = mean(res_speed$x), sd = sd(res_speed$x)))
# Q-Q normal for residual
qqplot.data(res_speed$x)
# From Box.test documentation:
# These tests are sometimes applied to the residuals from an ARMA(p, q) fit, 
# in which case the references suggest a better approximation to the null-hypothesis 
# distribution is obtained by setting fitdf = p+q, provided of course that lag > fitdf.
# In this case, for speed p=1 and q=1
Box.test(residuals(speed_model), type="Ljung-Box", fitdf=2, lag=144)



# Residual Histogram Direction
res_direc = as.data.frame(residuals(direc_model))
ggplot(res_direc, aes(x = x)) + 
  geom_histogram(aes(y = ..density..),
                 #bins=40,
                 fill = I("blue"),
                 col = I("blue"),
                 alpha=I(.5)
  ) + 
  xlab("Direction Model Residuals") +
  ylab("Frecuency") + 
  stat_function(fun = dnorm, args = list(mean = mean(res_direc$x), sd = sd(res_direc$x)))
# Q-Q normal for residual
qqplot.data(res_direc$x)
Box.test(residuals(direc_model), type="Ljung-Box", fitdf = 3, lag = 144)

# Unit root
plot(speed_model)

# Forecast 1 day 
speed_forecast <- forecast(speed_model, h = 144)
plot(speed_forecast, main="Speed Model")
sp_pred <- as.data.frame(speed_forecast)
sp_pred$timestamp <- test$timestamp

direc_forecast <- forecast(direc_model, h = 144)
plot(direc_forecast, main="Direction Model")
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
  geom_line(data=sp_pred, aes(x=timestamp, y=`Point Forecast`, colour="Forecast")) + #, color = "blue") + 
  geom_line(data=test, aes(x=timestamp, y=speed, colour="Testing")) + #, color = "red") + 
  scale_colour_manual("Legend", 
                      breaks = c("Forecast", "Testing"),
                      values = c("red", "blue")) +
  xlab("Timestamp") + 
  ylab("Speed")
speed_forecast_plot

# Direction
direc_forecast_plot <- ggplot() + 
  geom_line(data=di_pred, aes(x=timestamp, y=`Point Forecast`, colour="Forecast"))+ #, color = "blue") + 
  geom_line(data=test, aes(x=timestamp, y=direction, colour = "Testing")) + 
  scale_colour_manual("Legend", 
                      breaks = c("Forecast", "Testing"),
                      values = c("red", "blue")) +
  xlab("Timestamp") + 
  ylab("Direction")
direc_forecast_plot

### SIMULATION ###
color <- c(rep("#ff000010", 0), rep("#0000ff10", 100))
sim_speed <- simulate(speed_model, nsim = 144, future = TRUE)
sim_direc <- simulate(direc_model, nsim = 144, future = TRUE)
plot(sim_direc, type="l", col=color[1], xlab="h", ylab="Simulated Direction")
# Create simulated data for physical model 
for (sim in 2:50) { # 100 one day simulation
  sim_speed <- simulate(speed_model, nsim = 144, future = TRUE)
  sim_direc <- simulate(direc_model, nsim = 144, future = TRUE)
  #plot(sim_speed)
  #plot(sim_speed, type="l", col="#ff000010")
  lines(sim_direc, type="l", col=color[sim])
  #write.csv(x=sim_speed, file=paste("../data/simulated/speed/", sim, ".csv", sep = ""))
  #write.csv(x=sim_direc, file=paste("../data/simulated/direction/", sim, ".csv", sep = ""))
}
