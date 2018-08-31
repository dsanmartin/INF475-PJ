library('ggplot2')
library('forecast')
library('tseries')
theme_set(theme_gray())

# Read data
data <- read.csv("../data/originald08.csv", header=TRUE)
data <- data[, c(1, 5, 8)]
colnames(data) <- c("timestamp", "speed", "direction")
data$timestamp <- as.POSIXct(data$timestamp)

# Data description
summary(data[, 2:3])

# qplot(data$speed, 
#       geom = "histogram", 
#       xlab = "Speed",
#       fill = I("blue"),
#       col = I("blue"),
#       alpha=I(.2))


# Get data from 2017
data_2017 <- subset(data, data$timestamp >= "2017-01-01")

# 30 minutes frequency
data_2017_30 <- data_2017[seq(1, nrow(data_2017), 3), ]

# Timeseries
speed <- ts(data_2017_30$speed, frequency = 24*60/30)
direction <- ts(data_2017_30$direction, frequency = 24*60/30)

# # Timeserie plot
# p <- ggplot() + geom_line(data=data_2017_30, aes(x=timestamp, y=speed), color = "blue") + xlab("Timestamp") + ylab("Speed")
# p
# 
# # ACF and PACF
# Acf(speed)
# Pacf(speed)

# Models
speed_model <- auto.arima(speed)
speed_model

direc_model <- auto.arima(direction)
direc_model


