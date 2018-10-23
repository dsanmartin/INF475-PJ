library('ggplot2')
library('forecast')
library('tseries')
theme_set(theme_gray())

# Read data
data <- read.csv("../data/d05a.csv", header=TRUE)
data <- data[, c(1, 5, 8)] # Use datetime, speed and direction columns
colnames(data) <- c("timestamp", "speed", "direction")
data$timestamp <- as.POSIXct(data$timestamp, format='%Y-%m-%d %H:%M')

# All data description
summary(data[, 2:3])

# Data's histograms
qplot(data$speed,
      geom = "histogram",
      xlab = "",
      fill = I("blue"),
      col = I("blue"),
      alpha=I(.5))

qplot(data$direction,
      geom = "histogram",
      xlab = "Direction",
      fill = I("blue"),
      col = I("blue"),
      alpha=I(.5))

# Get january data
jan <- subset(data, data$timestamp >= "2018-01-01" & data$timestamp < "2018-02-01")
# Use first 30 days from January 2018 as train
train <- subset(data, data$timestamp >= "2018-01-01" & data$timestamp < "2018-01-31")
# Use day 31 from January 2018 as test
test <- subset(data, data$timestamp >= "2018-01-31" & data$timestamp < "2018-02-01")

# Remove data unused
rm(data)

# January data description
summary(jan[, 2:3])

# Histograms
qplot(jan$speed,
      geom = "histogram",
      xlab = "",
      fill = I("blue"),
      col = I("blue"),
      alpha=I(.5))

qplot(jan$direction,
      geom = "histogram",
      xlab = "",
      fill = I("blue"),
      col = I("blue"),
      alpha=I(.5))

# Timeseries plot
# Speed 
ggplot() + 
  geom_line(data=jan, aes(x=timestamp, y=speed), color = "blue") + 
  xlab("Timestamp") + 
  ylab("Speed")

# Direction
ggplot() + 
  geom_line(data=jan, aes(x=timestamp, y=direction), color = "blue") + 
  xlab("Timestamp") + 
  ylab("Direction")

# Save data (included in data folder)
#write.csv(file="../data/train_2018.csv", train, row.names=FALSE)
#write.csv(file="../data/test_2018.csv", test, row.names=FALSE)
