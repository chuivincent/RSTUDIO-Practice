#======================================================================================
#Linear Regression

#Linear regression is a very simple approach for supervised learning. Linear
#regression is a useful tool for predicting a quantitative response. 

#======================================================================================
# Packages
install.packages("dplyr")
install.packages("tidyverse")
install.packages("modelr")
install.packages("broom")
install.packages("ISLR")

library(tidyverse)  # data manipulation and visualization
library(modelr)     # provides easy pipeline modeling functions
library(broom)      # helps to tidy up model outputs
library(dplyr)
library(ISLR)
#======================================================================================
# Load data (remove row numbers included as X1 variable)
advertising <- read_csv("C:/Users/Vincent Chui/Desktop/RStudio/Learning/Advertising.csv") %>%
  select(-X1)

advertising

#A test set is a set of data used to assess the strength and utility of a predictive relationship. 
#conventional 60% / 40% split where we training our model on 60% of the data and then test the model 
#performance on 40% of the data that is withheld.

set.seed(123)
sample <- sample(c(TRUE, FALSE), nrow(advertising), replace = T, prob = c(0.6,0.4))
train <- advertising[sample, ]
test <- advertising[!sample, ]
#======================================================================================
#simple linear regression

#Y represents sales
#X represents TV advertising budget
#B0 is the intercept
#B1 is the coefficient (slope term) representing the linear relationship
#E is a mean-zero random error term

model1 <- lm(sales ~ TV, data = train)

summary(model1)
glance(model1)

tidy(model1)

# intercept estimate is 6.76 so when the TV advertising budget is zero we can expect sales to be 6,760 
# (remember we are using units of 1,000). And for every $1,000 increase in the TV advertising budget 
# we expect the average increase in sales to be 50 units.

confint(model1)

#Our results show us that our 95% confidence interval for B1(TV) is [.043, .057]. 
#Thus, since zero is not in this interval we can conclude that as the TV advertising budget 
#increases by $1,000 we can expect the sales to increase by 43-57 units. 

#This is also supported by the t-statistic provided by our results, which are computed by
#which measures the number of standard deviations that B1 is away from 0. 
#Thus a large t-statistic such as ours will produe a small p-value 
#(a small p-value indicates that it is unlikely to observe such a substantial association 
#between the predictor variable and the response due to chance). 
#Thus, we can conclude that a relationship between TV advertising budget and sales exists.

t.test(advertising$TV)
#======================================================================================
#Acessing Model Accuracy
#we want to understand the extent to which the model fits the data. This is typically referred 
#to as the goodness-of-fit. We can measure this quantitatively by assessing three things:
# 1. Residual standard error
# 2. R squared (R^2)
# 3. F-statistic


#RSE
#The RSE is an estimate of the standard deviation of E
#Roughly speaking, it is the average amount that the response will deviate from the true regression line

summary(model1)
#can get this value directly by the following
sigma(model1)
#An RSE value of 3.2 means the actual sales in each market will deviate from the true regression line 
#by approximately 3,200 units, on average. Is this significant? this is subjective but when compared 
#to the average value of sales over all markets the percentage error is 22%

sigma(model1)/mean(train$sales)

#The RSE provides an absolute measure of lack of fit of our model to the data. But since it is measured 
#in the units of Y, it is not always clear what constitutes a good RSE. The R^2 statistic provides an 
#alternative measure of fit. It represents the proportion of variance explained and so it always takes 
#on a value between 0 and 1, and is independent of the scale of Y. R^2 is simply a function of residual 
#sum of squares (RSS) and total sum of squares (TSS)

#RSquared 
rsquare(model1, data = train)
# The result suggests that TV advertising budget can explain 64% of the variability in our sales data.

#As a side note, in a simple linear regression model the R^2 value will equal the squared correlation between 
#X and Y
cor(train$TV, train$sales)^2

#F-statistic

#F-statistic tests to see if at least one predictor variable has a non-zero coefficient. This becomes more 
#important once we start using multiple predictors as in multiple linear regression

#larger F-statistic will produce a statistically significant p-value (p<0.05). In our case we see at the 
#bottom of our summary statement that the F-statistic is 210.8 producing a p-value of p<2.2e-16

#Combined, our RSE, R^2, and F-statistic results suggest that our model has an ok fit, but we could 
#likely do better.

#======================================================================================

#Assessing the model visually

ggplot(train, aes(TV, sales)) +
  geom_point() +
  geom_smooth(method = "lm") +
  geom_smooth(se = FALSE, color = "red")

#Here we use geom_smooth(method = "lm") followed by geom_smooth(). This allows us to compare the 
#linearity of our model (blue line with the 95% confidence interval in shaded region) with a non-linear 
#LOESS model. Considering the LOESS(locally weighted scatterplot smoothing) smoother remains within the 
#confidence interval we can assume the linear trend fits the essence of this relationship. However, we 
#do note that as the TV advertising budget gets closer to 0 there is a stronger reduction in sales 
#beyond what the linear trend follows.

#First is a plot of residuals versus fitted values. This will signal two important concerns:

#Non-linearity: if a discernible pattern (blue line) exists then this suggests either non-linearity or 
#that other attributes have not been adequately captured. Our plot indicates that the assumption of linearity is fair.

#Heteroskedasticity: an important assumption of linear regression is that the error terms have a constant variance, 
#Var(E)=sigma^2. If there is a funnel shape with our residuals, as in our plot, then we have violated this assumption. 
#Sometimes this can be resolved with a log or square root transformation of Y in our model.

# add model diagnostics to our training data
model1_results <- augment(model1, train)

ggplot(model1_results, aes(.fitted, .resid)) +
  geom_ref_line(h = 0) +
  geom_point() +
  geom_smooth(se = FALSE) +
  ggtitle("Residuals vs Fitted")

#The first is comparing standardized residuals versus fitted values. This is the same plot as above but with the residuals 
#standardized to show where residuals deviate by 1, 2, 3+ standard deviations. This helps us to identify outliers that 
#exceed 3 standard deviations. 
p1 <- ggplot(model1_results, aes(.fitted, .std.resid)) +
  geom_ref_line(h = 0) +
  geom_point() +
  geom_smooth(se = FALSE) +
  ggtitle("Standardized Residuals vs Fitted")

#The second is the scale-location plot. This plot shows if residuals are spread equally along the ranges of predictors. 
#This is how you can check the assumption of equal variance (homoscedasticity). It is good if you see a horizontal line 
#with equally (randomly) spread points.
p2 <- ggplot(model1_results, aes(.fitted, sqrt(.std.resid))) +
  geom_ref_line(h = 0) +
  geom_point() +
  geom_smooth(se = FALSE) +
  ggtitle("Scale-Location")

gridExtra::grid.arrange(p1, p2, nrow = 1)

#A Q-Q plot plots the distribution of our residuals against the theoretical normal distribution. The closer the points 
#are to falling directly on the diagonal line then the more we can interpret the residuals as normally distributed. 
#If there is strong snaking or deviations from the diagonal line then we should consider our residuals non-normally distributed. 
#In our case we have a little deviation in the bottom left-hand side which likely is the concern we mentioned earlier that as 
#the TV advertising budget approaches 0 the relationship with sales appears to start veering away from a linear relationship.


qq_plot <- qqnorm(model1_results$.resid)
qq_plot <- qqline(model1_results$.resid)

#cook's distance & residuals versus leverage plot
#These plot helps us to find influential cases (i.e., subjects) if any. Not all outliers are influential in linear regression analysis. 
#Even though data have extreme values, they might not be influential to determine a regression line. That means, the results would not 
#be much different if we either include or exclude them from analysis. They follow the trend in the majority of cases and they do not 
#really matter; they are not influential. On the other hand, some cases could be very influential even if they look to be within a 
#reasonable range of the values. They could be extreme cases against a regression line and can alter the results if we exclude them 
#from analysis. Another way to put it is that they do not get along with the trend in the majority of the cases.

par(mfrow=c(1, 2))

plot(model1, which = 4, id.n = 5)
plot(model1, which = 5, id.n = 5)

model1_results %>%
  top_n(5, wt = .cooksd)
#======================================================================================
#Making Predictions

(test <- test %>% 
    add_predictions(model1))

#test MSE
#one that produces the lowest test sample MSE is the preferred model.
test %>% 
  add_predictions(model1) %>%
  summarise(MSE = mean((sales - pred)^2))

train %>% 
  add_predictions(model1) %>%
  summarise(MSE = mean((sales - pred)^2))
#======================================================================================

#Multiple Regression

advertising <- read_csv("C:/Users/Vincent Chui/Desktop/RStudio/Learning/Advertising.csv") %>%
  select(-X1)

advertising

model2 <- lm(sales ~ TV + radio + newspaper, data = train)

summary(model2)
#======================================================================================
#Assessing Coefficients
#The interpretation of our coefficients is the same as in a simple linear regression model. 
#First, we see that our coefficients for TV and Radio advertising budget are statistically significant 
#(p-value < 0.05) while the coefficient for Newspaper is not. Thus, changes in Newspaper budget do not 
#appear to have a relationship with changes in sales. 

#for TV our coefficent suggests that for every $1,000 increase in TV advertising budget, holding all other 
#predictors constant, we can expect an increase of 47 sales units, on average

#Radio coefficient suggests that for every $1,000 increase in Radio advertising budget, holding all other 
#predictors constant, we can expect an increase of 196 sales units, on average

tidy(model2)


#confidence interval
confint(model2)

model1 <- lm(sales ~ TV, data = train)
#======================================================================================
#assessing model accuracy

list(model1 = broom::glance(model1), model2 = broom::glance(model2))

#Assessing model accuracy is very similar as when assessing simple linear regression models. Rather than 
#repeat the discussion, here I will highlight a few key considerations. First, multiple regression is when the 
#F-statistic becomes more important as this statistic is testing to see if at least one of the coefficients is 
#non-zero. When there is no relationship between the response and predictors, we expect the F-statistic to take 
#on a value close to 1. On the other hand, if at least predictor has a relationship then we expect 
#F>1. In our summary print out above for model 2 we saw that F=445.9 with p<0.05 suggesting that at least 
#one of the advertising media must be related to sales.

#R^2: Model 2's R^2=.92 is substantially higher than model 1 suggesting that model 2 does a better job explaining 
#the variance in sales. It iss also important to consider the adjusted R^2. The adjusted R^2 is a modified version of R^2
#that has been adjusted for the number of predictors in the model. The adjusted R^2 increases only if the new term 
#improves the model more than would be expected by chance. Thus, since model 2's adjusted R^2 is also substantially 
#higher than model 1 we confirm that the additional predictors are improving the models performance.

#RSE: Model 2's RSE (sigma) is lower than model 1. This shows that model 2 reduces the variance of our ϵ parameter 
#which corroborates our conclusion that model 2 does a better job modeling sales.

#F-statistic: the F-statistic (statistic) in model 2 is larger than model 1. Here larger is better and suggests 
#that model 2 provides a better goodness-of-fit.

#Other: We can also use other various statistics to compare the quality of our models. These include Akaike 
#information criterion (AIC) and Bayesian information criterion (BIC), which we see in our results, among others. 
#models with lower AIC and BIC values are considered of better quality than models with higher values.

#add model diagnostics to our training data

#First, if we compare model 2's residuals versus fitted values we see that model 2 has reduced concerns with 
#heteroskedasticity; however, we now have discernible patter suggesting concerns of linearity. 

model1_results <- model1_results %>%
  mutate(Model = "Model 1")

model2_results <- augment(model2, train) %>%
  mutate(Model = "Model 2") %>%
  rbind(model1_results)

ggplot(model2_results, aes(.fitted, .resid)) +
  geom_ref_line(h = 0) +
  geom_point() +
  geom_smooth(se = FALSE) +
  facet_wrap(~ Model) +
  ggtitle("Residuals vs Fitted")

#concern with normality is supported when we compare the Q-Q plots. So although our model is performing better 
#numerically, we now have a greater concern with normality then we did before! 

par(mfrow=c(1, 2))

# Left: model 1
qqnorm(model1_results$.resid); qqline(model1_results$.resid)

# Right: model 2
qqnorm(model2_results$.resid); qqline(model2_results$.resid)
#======================================================================================
#Making Predictions

#how our models compare when making predictions on an out-of-sample data set we'll compare MSE. Here we can use 
#gather_predictions to predict on our test data with both models and then, as before, compute the MSE. Here we see 
#that model 2 drastically reduces MSE on the out-of-sample. So although we still have lingering concerns over residual 
#normality model 2 is still the preferred model so far.

test %>%
  gather_predictions(model1, model2) %>%
  group_by(model) %>%
  summarise(MSE = mean((sales-pred)^2))

#incorporating interactions

#In our previous analysis of the Advertising data, we concluded that both TV and radio seem to be associated with sales. 
#The linear models that formed the basis for this conclusion assumed that the effect on sales of increasing one advertising 
#medium is independent of the amount spent on the other media. For example, the linear model (Eq. 10) states that the average
#effect on sales of a one-unit increase in TV is always β1, regardless of the amount spent on radio.

#However, this simple model may be incorrect. Suppose that spending money on radio advertising actually increases the effectiveness 
#of TV advertising, so that the slope term for TV should increase as radio increases. In this situation, given a fixed budget of 
#$100,000, spending half on radio and half on TV may increase sales more than allocating the entire amount to either TV or to radio. 
#In marketing, this is known as a synergy effect, and in statistics it is referred to as an interaction effect. One way of extending 
#our model 2 to allow for interaction effects is to include a third predictor, called an interaction term, which is constructed by 
#computing the product of X1 and X2 (here we'll drop the Newspaper variable). This results in the model
#Y=b0+b1X1+b2X2+b3X1X2+e

#Now the effect of X1 on Y is no longer constant as adjusting X2 will change the impact of X1 on Y. We can interpret β3 as the increase 
#in the effectiveness of TV advertising for a one unit increase in radio advertising (or vice-versa). To perform this in R we can use 
#either of the following. Note that option B is a shorthand version as when you create the interaction effect with *, R will automatically 
#retain the main effects.

# option A
model3 <- lm(sales ~ TV + radio + TV * radio, data = train)

# option B
model3 <- lm(sales ~ TV * radio, data = train)

tidy(model3)

#Assessing Coefficients
#We see that all our coefficients are statistically significant. Now we can interpret this as an increase in TV advertising of $1,000 is 
#associated with increased sales of Radio. And an increase in Radio advertising of $1,000 will be associated with 
#an increase in sales of TV.
#======================================================================================
# Model Accuracy
#We can compare our model results across all three models. We see that our adjusted R^2 and F-statistic are highest with model 3 and our 
#RSE, AIC, and BIC are the lowest with model 3; all suggesting the model 3 out performs the other models.


list(model1 = broom::glance(model1), 
     model2 = broom::glance(model2),
     model3 = broom::glance(model3))

#Assessing Our Model Visually
#Visually assessing our residuals versus fitted values we see that model three does a better job with constant variance and, with the 
#exception of the far left side, does not have any major signs of non-normality.

# add model diagnostics to our training data
model3_results <- augment(model3, train) %>%
  mutate(Model = "Model 3") %>%
  rbind(model2_results)

ggplot(model3_results, aes(.fitted, .resid)) +
  geom_ref_line(h = 0) +
  geom_point() +
  geom_smooth(se = FALSE) +
  facet_wrap(~ Model) +
  ggtitle("Residuals vs Fitted")

#As an alternative to the Q-Q plot we can also look at residual histograms for each model. Here we see that model 3 has a couple large 
#left tail residuals. These are related to the left tail dip we saw in the above plots.

ggplot(model3_results, aes(.resid)) +
  geom_histogram(binwidth = .25) +
  facet_wrap(~ Model, scales = "free_x") +
  ggtitle("Residual Histogram")

#These residuals can be tied back to when our model is trying to predict low levels of sales (< 10,000). If we remove these 
#sales our residuals are more normally distributed. What does this mean? Basically our linear model does a good job predicting 
#sales over 10,000 units based on TV and Radio advertising budgets; however, the performance deteriates when trying to predict 
#sales less than 10,000 because our linear assumption does not hold for this segment of our data.

model3_results %>%
  filter(sales > 10) %>%
  ggplot(aes(.resid)) +
  geom_histogram(binwidth = .25) +
  facet_wrap(~ Model, scales = "free_x") +
  ggtitle("Residual Histogram")

#This can be corroborated by looking at the Cook's Distance and Leverage plots. Both of them highlight observations 
#3, 5, 47, 65, and 94 as the top 5 influential observations.

par(mfrow=c(1, 2))

plot(model3, which = 4, id.n = 5)
plot(model3, which = 5, id.n = 5)

#If we look at these observations we see that they all have low Sales levels.

train[c(3, 5, 47, 65, 94),]

#======================================================================================

#Making Predictions
#Again, to see how our models compare when making predictions on an out-of-sample data set we’ll compare the MSEs across all our models. 
#Here we see that model 3 has the lowest out-of-sample MSE, further supporting the case that it is the best model and has not overfit 
#our data.

test %>%
  gather_predictions(model1, model2, model3) %>%
  group_by(model) %>%
  summarise(MSE = mean((sales-pred)^2))

#Additional Considerations

#Qualitative Predictors
#In our discussion so far, we have assumed that all variables in our linear regression model are quantitative. But in practice, this is 
#not necessarily the case; often some predictors are qualitative. For example, the Credit data set records balance (average credit card 
#debt for a number of individuals) as well as several quantitative predictors: age, cards (number of credit cards), education 
#(years of education), income (in thousands of dollars), limit (credit limit), and rating (credit rating).

#Suppose that we wish to investigate differences in credit card balance between males and females, ignoring the other variables for the moment. 
#If a qualitative predictor (also known as a factor) only has two levels, or possible values, then incorporating it into a regression model is 
#very simple. We simply create an indicator or dummy variable that takes on two possible numerical values. For example, based on the gender, 
#we can create a new variable that takes the form

#1 if ith person is male
#0 if ith person is female

#and use this variable as a predictor in the regression equation. This results in the model

#β0+β1+ϵi if ith person is male
#β0+ϵi if ith person is female

credit <- read_csv("http://www-bcf.usc.edu/~gareth/ISL/Credit.csv")
model4 <- lm(Balance ~ Gender, data = credit)

tidy(model4)
#The results below suggest that females are estimated to carry $529.54 in credit card debt where males carry $529.54 - $19.73 = $509.81.

#The decision to code females as 0 and males as 1 in is arbitrary, and has no effect on the regression fit, but does alter the interpretation
#of the coefficients. If we want to change the reference variable (the variable coded as 0) we can change the factor levels.

credit$Gender <- factor(credit$Gender, levels = c("Male", "Female"))

lm(Balance ~ Gender, data = credit) %>%
  tidy()

#A similar process ensues for qualitative predictor categories with more than two levels. For instance, if we want to assess the impact that 
#ethnicity has on credit balance we can run the following model. Ethnicity has three levels: African American, Asian, Caucasian. We interpret 
#the coefficients much the same way. In this case we see that the estimated balance for the baseline, African American, is $531.00. It is 
#estimated that the Asian category will have $18.69 less debt than the African American category, and that the Caucasian category will have 
#$12.50 less debt than the African American category. However, the p-values associated with the coefficient estimates for the two dummy variables 
#are very large, suggesting no statistical evidence of a real difference in credit card balance between the ethnicities.

lm(Balance ~ Ethnicity, data = credit) %>%
  tidy

#The process for assessing model accuracy, both numerically and visually, along with making and measuring predictions can follow the same process 
#as outlined for quantitative predictor variables.

#Transformations

#Linear regression models assume a linear relationship between the response and predictors. But in some cases, the true relationship between the 
#response and the predictors may be non-linear. We can accomodate certain non-linear relationships by transforming variables 
#(i.e. log(x), sqrt(x)) or using polynomial regression.


#As an example consider the Auto data set. We can see that a linear trend does not fit the relationship between mpg and horsepower.

auto <- ISLR::Auto

ggplot(auto, aes(horsepower, mpg)) +
  geom_point() +
  geom_smooth(method = "lm")

#We can try to address the non-linear relationship with a quadratic relationship, which takes the form of:
#mpg=β0+β1×horsepower+β2×horsepower2+ε


model5 <- lm(mpg ~ horsepower + I(horsepower^2), data = auto)

tidy(model5)

ggplot(auto, aes(horsepower, mpg)) +
  geom_point() +
  geom_smooth(method = "lm", formula = y ~ x + I(x^2))


#Correlation of Error Terms

#An important assumption of the linear regression model is that the error terms, ϵ1,ϵ2,…,ϵn, are uncorrelated. Correlated residuals
#frequently occur in the context of time series data, which consists of observations for which measurements are obtained at discrete 
#points in time. In many cases, observations that are obtained at adjacent time points will have positively correlated errors. This 
#will result in biased standard errors and incorrect inference of model results.

#To illustrate, we'll create a model that uses the number of unemployed to predict personal consumption expenditures (using the economics 
#data frame provided by ggplot2). The assumption is that as more people become unemployed personal consumption is likely to reduce. However, 
#if we look at our model's residuals we see that adjacent residuals tend to take on similar values. In fact, these residuals have a .998 
#autocorrelation. This is a clear violation of our assumption. 

df <- economics %>% 
  mutate(observation = 1:n())

model6 <- lm(pce ~ unemploy, data = df)

df %>%
  add_residuals(model6) %>%
  ggplot(aes(observation, resid)) +
  geom_line()

#Collinearity

#Collinearity refers to the situation in which two or more predictor variables are closely related to one another. The presence of collinearity 
#can pose problems in the regression context, since it can be difficult to separate out the individual effects of collinear variables on the 
#response. In fact, collinearity can cause predictor variables to appear as statistically insignificant when in fact they are significant.

#For example, compares the coefficient estimates obtained from two separate multiple regression models. The first is a regression of balance on 
#age and limit, and the second is a regression of balance on rating and limit. In the first regression, both age and limit are highly significant 
#with very small p- values. In the second, the collinearity between limit and rating has caused the standard error for the limit coefficient estimate 
#to increase by a factor of 12 and the p-value to increase to 0.701. In other words, the importance of the limit variable has been masked due to 
#the presence of collinearity.

model7 <- lm(Balance ~ Age + Limit, data = credit)
model8 <- lm(Balance ~ Rating + Limit, data = credit)

list(`Model 1` = tidy(model7),
     `Model 2` = tidy(model8))

#A simple way to detect collinearity is to look at the correlation matrix of the predictors. An element of this matrix that is large in absolute 
#value indicates a pair of highly correlated variables, and therefore a collinearity problem in the data. Unfortunately, not all collinearity problems 
#can be detected by inspection of the correlation matrix: it is possible for collinear- ity to exist between three or more variables even if no pair 
#of variables has a particularly high correlation. We call this situation multicollinearity.

#Instead of inspecting the correlation matrix, a better way to assess multi- collinearity is to compute the variance inflation factor (VIF). The VIF 
#is the ratio of the variance of βj when fitting the full model divided by the variance of βj if fit on its own. The smallest possible value for VIF 
#is 1, which indicates the complete absence of collinearity. Typically in practice there is a small amount of collinearity among the predictors. As a 
#rule of thumb, a VIF value that exceeds 5 or 10 indicates a problematic amount of collinearity.

#We can use the vif function from the car package to compute the VIF. As we see below model 7 is near the smallest possible VIF value where model 8 
#has obvious concerns.

car::vif(model7)
car::vif(model8)

