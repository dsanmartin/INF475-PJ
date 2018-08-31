library('ggplot2')
library('tseries')
theme_set(theme_gray())

# Read data
data <- read.csv("../data/originald08.csv", header=TRUE)
data <- data[, c(1, 5, 8)]
colnames(data) <- c("timestamp", "speed", "direction")
data$timestamp <- as.POSIXct(data$timestamp)

# Data description
summary(data[, 2:3])

# Histograms
qplot(data$speed,
      geom = "histogram",
      xlab = "Speed",
      fill = I("blue"),
      col = I("blue"),
      alpha=I(.5))

qplot(data$direction,
      geom = "histogram",
      xlab = "Direction",
      fill = I("blue"),
      col = I("blue"),
      alpha=I(.5))

# Get data from 2017
data_2017 <- subset(data, data$timestamp >= "2017-01-01")

# 30 minutes frequency
data_2017_30 <- data_2017[seq(1, nrow(data_2017), 3), ]

# Timeserie plot
pspeed <- ggplot() + geom_line(data=data_2017_30, aes(x=timestamp, y=speed), color = "blue") + xlab("Timestamp") + ylab("Speed")
pspeed

# ACF and PACF
speed_ACF <- Acf(data_2017_30$speed, plot = FALSE)
plot(speed_ACF, main = "Speed Autocorrelation")
speed_PACF <- Pacf(data_2017_30$speed, plot = FALSE)
plot(speed_PACF, main = "Speed Partial Autocorrelation")

# Timeserie plot
pdirec <- ggplot() + geom_line(data=data_2017_30, aes(x=timestamp, y=direction), color = "blue") + xlab("Timestamp") + ylab("Direction")
pdirec

# ACF and PACF
direc_ACF <- Acf(data_2017_30$direction, plot = FALSE)
plot(direc_ACF, main = "Direction Autocorrelation")
direc_pacf <- Pacf(data_2017_30$direction, plot = FALSE)
plot(direc_pacf, main = "Direction Partial Autocorrelation")

# Save data
write.csv(file="../data/2017_30min.csv", x=data_2017_30)