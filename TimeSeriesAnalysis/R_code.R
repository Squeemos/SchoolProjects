# install.packages('forecast',dependencies = TRUE)
# install.packages('ggplot2',dependencies = TRUE)
# install.packages('tseries')
# install.packages('ggfortify')

library(forecast)
library(ggplot2)
library(ggfortify)
library(tseries)
library(uroot)

train <- read.csv("./dataset/adjusted_training.csv")
test <- read.csv("./dataset/DailyDelhiClimateTest.csv")

data <- ts(train['meantemp'],start=2013,frequency=365)
test_data <- ts(test['meantemp'],start=2017,frequency=365)

autoplot(data)
gghistogram(data)
ggAcf(data)
ggPacf(data)

# This will find our lag period and trend for later
data %>% mstl() %>% autoplot()

data %>% kpss.test(lshort=FALSE)
data %>% kpss.test(null="Trend",lshort=FALSE)
data %>% pp.test(lshort=FALSE)
data %>% adf.test(k=365)
data %>% ocsb.test(maxlag=365) %>% summary()
data %>% hegy.test()
data %>% adf.test()

data %>% ch.test() %>% summary()

for (type in c("trend", "level"))
{
  print(type)
  for (val in c("kpss","adf","pp"))
  {
    print(val)
    print(ndiffs(data,test=val,type=type))
  }
}

for (val in c("seas", "ocsb", "hegy", "ch"))
{
  print(val)
  print(nsdiffs(data,test=val))
}

# Look at the differed data
diff <- diff(data)
autoplot(diff)
ggAcf(diff)
ggPacf(diff)
nsdiffs(diff)
ndiffs(diff)

View(data)

# Make our Arima model
model <- auto.arima(data)
model <-Arima(data,order=c(1,1,1),seasonal=c(0,1,0))

# Forecast compared to the future
forecast(model,h=114) %>% autoplot(xlim=c(2016.5,2017.5)) + autolayer(test_data)

Box.test(model$residuals,lag=5,type="Box-Pierce")

model

checkresiduals(model)
accuracy(model)

# This is all compared to the training data, what about the testing? (doesn't work now for some reason)
# test_model <- Arima(test_data, model=model)
# accuracy(test_model)
