#======================================================================================
#Time Series Analysis
install.packages("ggplot2")
install.packages("ggfortify")
install.packages("forecast")
install.packages("fpp2")
install.packages("tseries")


library(forecast)
library(fpp2)
library(tseries)
library(ggplot2)
library(ggfortify)
#======================================================================================
data(AirPassengers)
AP <- AirPassengers
# Take a look at the class of the dataset AirPassengers
class(AP)

# Take a look at the entries
AP

# Check for missing values
sum(is.na(AP))

# Check the frequency of the time series
frequency(AP)

# Check the cycle of the time series
cycle(AP)

# Review the table summary
summary(AP)

# Plot the raw data using the base plot function
plot(AP,xlab="Date", ylab = "Passenger numbers (1000's)",main="Air Passenger numbers from 1949 to 1961")


#As an alternative to the base plot function, so we can also use the extension ggfortify R package from the 
#ggplot2 package, to plot directly from a time series. The benefits are not having to convert to a dataframe 
#as required with ggplot2, but still having access to the layering grammar of graphics.
autoplot(AP) + labs(x ="Date", y = "Passenger numbers (1000's)", title="Air Passengers from 1949 to 1961") 

#use the boxplot function to see any seasonal effects
boxplot(AP~cycle(AP),xlab="Date", ylab = "Passenger Numbers (1000's)" ,main ="Monthly Air Passengers Boxplot from 1949 to 1961")

#From these exploratory plots, we can make some initial inferences:

#The passenger numbers increase over time with each year which may be indicative of an increasing linear trend, perhaps due to 
#increasing demand for flight travel and commercialisation of airlines in that time period.

#In the boxplot there are more passengers travelling in months 6 to 9 with higher means and higher variances than the other months, 
#indicating seasonality with a apparent cycle of 12 months. The rationale for this could be more people taking holidays and fly 
#over the summer months in the US.

#AirPassengers appears to be multiplicative time series as the passenger numbers increase, it appears so does the pattern of seasonality.

#There do not appear to be any outliers and there are no missing values. Therefore no data cleaning is required.
#======================================================================================
#TIME SERIES DECOMPOSITION
#We will decompose the time series for estimates of trend, seasonal, and random components using moving average method.

#The multiplicative model is:
#    Y[t]=T[t]*S[t]*e[t]

#where

#Y(t) is the number of passengers at time t,
#T(t) is the trend component at time t,
#S(t) is the seasonal component at time t,
#e(t) is the random error component at time t.

decomposeAP <- decompose(AP,"multiplicative")
autoplot(decomposeAP)
#======================================================================================
#TEST STATIONARITY OF THE TIME SERIES
#A stationary time series has the conditions that the mean, variance and covariance are not functions of time. In order to fit arima models, 
#the time series is required to be stationary. We will use two methods to test the stationarity.

#1. Test stationarity of the time series (ADF)
#In order to test the stationarity of the time series, let's run the Augmented Dickey-Fuller Test using the adf.test function from the tseries R package.

#First set the hypothesis test:
  
#The null hypothesis H0 : that the time series is non stationary
#The alternative hypothesis HA : that the time series is stationary

adf.test(AP) 

#As a rule of thumb, where the p-value is less than 5%, we strong evidence against the null hypothesis, so we reject the null hypothesis. As per the test 
#results above, the p-value is 0.01 which is <0.05 therefore we reject the null in favour of the alternative hypothesis that the time series is stationary.

#Test stationarity of the time series (Autocorrelation)

#Another way to test for stationarity is to use autocorrelation. We will use autocorrelation function (acf) in from the base stats R package. This function 
#plots the correlation between a series and its lags ie previous observations with a 95% confidence interval in blue. If the autocorrelation crosses the 
#dashed blue line, it means that specific lag is significantly correlated with current series.

autoplot(acf(AP,plot=FALSE))+ labs(title="Correlogram of Air Passengers from 1949 to 1961") 

#The maximum at lag 1 or 12 months, indicates a positive relationship with the 12 month cycle.

#Since we have already created the decomposeAP list object with a random component, we can plot the acf of the decomposeAP$random.

# Review random time series for any missing values
decomposeAP$random 

# Autoplot the random time series from 7:138 which exclude the NA values
autoplot(acf(decomposeAP$random[7:138],plot=FALSE))+ labs(title="Correlogram of Air Passengers Random Component from 1949 to 1961") 

#We can see that the acf of the residuals is centered around 0.

#======================================================================================
#FIT A TIME SERIES MODEL

#Linear Model

#Since there is an upwards trend we will look at a linear model first for comparison. We plot AirPassengers raw dataset with a blue linear model.

autoplot(AP) + geom_smooth(method="lm")+ labs(x ="Date", y = "Passenger numbers (1000's)", title="Air Passengers from 1949 to 1961") 
#This may not be the best model to fit as it doesn’t capture the seasonality and multiplicative effects over time.

#======================================================================================
# ARIMA Model

#Use the auto.arima function from the forecast R package to fit the best model and coefficients, given the default parameters including seasonality 
#as TRUE. Note we have used the ARIMA modeling procedure as referenced

arimaAP <- auto.arima(AP)
arimaAP

#The ARIMA(2,1,1)(0,1,0)[12] model parameters are lag 1 differencing (d), an autoregressive term of second lag (p) and a moving average model of order 1 (q). 
#Then the seasonal model has an autoregressive term of first lag (D) at model period 12 units, in this case months.

#The ARIMA fitted model is:
#    Y=0.5960Yt−2+0.2143Yt−12−0.9819et−1+E

#where E is some error.

#The ggtsdiag function from ggfortify R package performs model diagnostics of the residuals and the acf. will include a autocovariance plot.

ggtsdiag(arimaAP)
#The residual plots appear to be centered around 0 as noise, with no pattern. the arima model is a fairly good fit.
#======================================================================================
# CALCULATE FORECASTS

#Finally we can plot a forecast of the time series using the forecast function, again from the forecast R package, with a 95% confidence interval 
#where h is the forecast horizon periods in months.

forecastAP <- forecast(arimaAP, level = c(95), h = 36)
autoplot(forecastAP)

#Model Identification and Estimation
#We need to find the appropriate values of p,d,q representing the AR order, the degress of differencing, and the MA order respectively. We will use 
#auto.arima to find the best ARIMA model to univariate time series (i.e. a time series that consists of single (scalar) observations recorded sequentially 
#over equal time increments.)
findbest <-auto.arima(AP)
findbest

plot(forecast(findbest,h=36))

#Create ARIMA prediction model
fit <- arima(AirPassengers, order=c(0, 1, 1), list(order=c(0, 1, 0), period = 12))
fit

#Compute prediction intervals of 95% confidence level for each prediction

fore <- predict(fit, n.ahead=24)
# calculate upper (U) and lower (L) prediction intervals
U <- fore$pred + 2*fore$se # se: standard error (quantile is 2 as mean=0)
L <- fore$pred - 2*fore$se
# plot observed and predicted values
ts.plot(AirPassengers, fore$pred, U, L, col=c(1, 2, 4, 4), lty=c(1, 1, 2, 2))
library(graphics)
legend("topleft", c("Actual", "Forecast", "Error Bounds (95% prediction interval)"), 
       col=c(1, 2, 4),lty=c(1, 1, 2))

#In the above figure, the red solid line shows the forecasted values, and the blue dotted lines are error bounds at a confidence level of 95%.
#======================================================================================
#Residual Analysis
#The sample autocorrelation function (ACF) for a series gives correlations between the series xt and lagged values of the series for lags of 1, 2, 3, and so on. 
#The lagged values can be written as xt-1, xt-2, xt-3,and so on. The ACF gives correlations between xt and xt-1, xt and xt-2, and so on. The ideal for a sample 
#ACF of residuals is that there aren’t any significant correlations for any lag.

#The partial autocorrelation function (PACF) plays an important role in data analyses aimed at identifying the extent of the lag in an autoregressive model. 
#By plotting the partial autocorrelative functions one could determine the appropriate lags p in an AR (p) model or in an extended ARIMA (p,d,q) model.


res <- residuals(fit)  # get residuals from fit
# check acf and pacf of residuals
acf(res)
pacf(res)


#The above figure shows the ACF of the residuals for a model. The “lag” (time span between observations) is shown along the horizontal, and the autocorrelation 
#is on the vertical. The red lines indicated bounds for statistical significance. This is a good ACF for residuals. Nothing is significant; that’s what we want 
#for residuals.

#The lag at which the PACF cuts off is the indicated number of AR terms (i.e. p).
#The lag at which the ACF cuts off is the indicated number of MA terms (i.e. q)

#Check normality using Q-Q plot
#arima() fits the model using maximum likelihood estimation (assuming Gaussian residual series) Now, plot the Q–Q plots, which measures the agreement of a fitted 
#distribution with observed data…

# qqnorm is a generic function the default method of which produces a normal QQ plot of the values in y. qqline adds a line to a “theoretical”, by default normal, 
#quantile-quantile plot which passes through the probs quantiles, by default the first and third quartiles.
qqnorm(residuals(fit))
qqline(residuals(fit))

#The linearity of the points suggests that the data are normally distributed with mean = 0.

#Test for stationarity using ADF test
#To cross check the stationarity of timeseries, use Augemented Dickey-Fuller test (from tseries package). Rejecting the null hypothesis suggests that a time series 
#is stationary

adf.test(fit$residuals, alternative ="stationary")

#From the above p-value, we can conclude that the residuals of our ARIMA prediction model is stationary.
