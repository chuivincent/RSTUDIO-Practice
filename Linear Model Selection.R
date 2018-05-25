#======================================================================================
#Linear Model Selection

#Often times some or many of the variables used in a multiple regression model are
#not associated with the response variable. Including such irrelevant variables will
#lead to unnecessary complexity in the resulting model. Manually filtering through
#and comparing regression models is too tedious and this there are several approaches
#that exist to automatically perform feature/variable selection which will provide
#the best regression.

#In this example we will be using a data set that contains information regarding 322
#MLB players.

#======================================================================================

#======================================================================================
#Packages
install.packages("tidyverse") # data manipulation and visualization
install.packages("leaps") # model selection functions

library(tidyverse)
library(leaps)
#======================================================================================
#load data and remove rows with missing data

(hitters <- na.omit(ISLR::Hitters) %>%
    as_tibble)
#======================================================================================

#Best Subset Selection

#To perform best subset selection, we fit a separate least squares regression for each 
#possible combination of the p predictors. We fit all p models that contain exactly one 
#predictor, all (P c 2)=p(p-1)/2 models that contain exactly two predictors,and so forth. 
#We then look at all of the resulting models, with the goal of identifying the one that is best.

#The three-stage process of performing best subset selection includes:

#Step 1: Let M0 denote the null model, which contains no predictors. This model simply predicts
#        the sample mean for each observation.

#Step 2: For k=1,2,...p:Fit all (p c k) models that contain exactly k predictors.Pick the best among 
#        these (p c k) models, and call it Mk. Here best is defined as having the smallest 
#        RSS(residual sum of squares), or equivalently largest R^2.

#Step 3: Select a single best model from among M0,....,Mp using cross-validated prediction error, 
#        Cp, AIC(Akaike information criterion), BIC(Bayesian information criterion), or adjusted R^2

#We can perform a best subset search using regsubsets (part of the leaps library), which identifies 
#the best model for a given number of k predictors, where best is quantified using RSS. The syntax
#is the same as the lm (linear model) function. By default, regsubsets only reports results up to 
#the best eight-variable model but, the nvmax option can be used in order to return as many variables 
#as are desired however here we fit up to a 19-variable model.

best_subset <- regsubsets(Salary ~ ., hitters, nvmax = 19)

#The resubsets function returns a list-object with lots of information. By using the summary command, we can
#assess the best set of variables for each model size. For a model with 1 variable we see that CRBI has an 
#asterisk signalling that a regressionmodel with Salary ~ CRBI is the best single variable model. The best
#2 variable model is Salary ~ CRBI + Hits and the best 3 variable model is Salary ~ CRBI + Hits + PutOuts and 
#so forth.

summary(best_subset)


#We can also get get the RSS, R^2, adjusted R^2, Cp, and BIC from the results which helps us to assess 
#the best overall model;however, we will illustrate this in the comparing models section. First, let's 
#look at how to perform stepwise selection.

#======================================================================================
#Stepwise selection

#For computational reasons, best subset selection cannot be applied when the number of p predictor variables is large. Best subset 
#selection may also suffer from statistical problems when p is large. The larger the search space, the higher the chance of finding 
#models that look good on the training data, even though they might not have any predictive power on future data. Thus an enormous 
#search space can lead to overfitting and high variance of the coefficient estimates. For both of these reasons, stepwise methods, 
#which explore a far more restricted set of models, are attractive alternatives to best subset selection.
#======================================================================================
#Foreward Stepwise

#Forward stepwise selection begins with a model containing no predictors, and then adds predictors to the model, one-at-a-time, 
#until all of the predictors are in the model. At each step the variable that gives the greatest additional 
#improvement to the fit is added to the model.

#The three-stage process of performing forward stepwise selection includes:
  
#Step 1: Let M0 denote the null model, which contains no predictors. This model simply predicts the sample mean for each observation.

#Step 2: For k=0,...,p-1:
#        Consider all p-k models that augment the predictors in Mk with one additional predictor.
#        Choose the best among these p-k models, and call it Mk+1. Here best is defined as having smallest RSS or highest R^2.

#Step 3: Select a single best model from among M0,....,Mp using cross-validated prediction error, Cp, AIC, BIC, or adjusted R^2

#We can perform forward stepwise using regsubsets by setting method = "forward"

forward <- regsubsets(Salary ~ ., hitters, nvmax = 19, method = "forward")


#======================================================================================
#Backward Stepwise

#Backward stepwise selection provides an efficient alternative to best subset selection. However, unlike forward stepwise 
#selection, it begins with the full least squares model containing all p predictors, and then iteratively removes the least 
#useful predictor, one-at-a-time.

#The three-stage process of performing forward stepwise selection includes:

#Step 1: Let Mp denote the full model, which contains all p predictors.

#Step 2: For k=p,p-1,...,1:
#        Consider all k models that contain all but one of the predictors in Mk, for a total of k-1 predictors.
#        Choose the best among the k models, and call it Mk-1. Here best is defined as having smallest RSS or highest R^2.

#Step 3: Select a single best model from among M0,...,M pusing cross-validated prediction error, Cp, AIC, BIC, or adjusted R^2

#We can perform backward stepwise using regsubsets by setting method = "backward"

backward <- regsubsets(Salary ~ ., hitters, nvmax = 19, method = "backward")

#======================================================================================
#Comparing Models

#So far,it has been demonstrated how to perform the best subset and stepwise procedures. Perform step 3 discussed in each of the 3-stage 
#processes outlined above.

#In order to select the best model with respect to test error, we need to estimate this test error. There are two common approaches

# 1.We can indirectly estimate test error by making an adjustment to the training error to account for the bias due to overfitting.
# 2.We can directly estimate the test error, using either a validation set approach or a cross-validation approach.

#======================================================================================

#Indirectly Estimating Test Error with Cp, AIC, BIC, and Adjusted R^2

#When performing the best subset or stepwise approaches, the M0,..,Mp models selected are selected based on the fact that they 
#minimize the training set mean square error (MSE). Because of this and the fact that using the training MSE and R^2
#will bias our results we should not use these statistics to determine which of the M0,..,Mp models is "the best".

#However, a number of techniques for adjusting the training error for the model size are available. These approaches can be 
#used to select among a set of models with different numbers of variables. 

#Cp -> Minimize
#Akaike information criterion (AIC) -> Minimize
#Bayesian information criterion (BIC) -> Minimize
#adjusted R^2 -> Maximize

#Each of these statistics adds a penalty to the training RSS in order to adjust for the fact that the training error tends to 
#underestimate the test error. Clearly, the penalty increases as the number of predictors in the model increases.

#Therefore, these statistics provide an unbiased estimate of test MSE. If we perform our model using a training vs. testing 
#validation approach we can use these statistics to determine the preferred model. These statistics are contained in the output 
#provided by the regsubsets function. 

# create training - testing data
set.seed(1)
sample <- sample(c(TRUE, FALSE), nrow(hitters), replace = T, prob = c(0.6,0.4))
train <- hitters[sample, ]
test <- hitters[!sample, ]

# perform best subset selection
best_subset <- regsubsets(Salary ~ ., train, nvmax = 19)
results <- summary(best_subset)

# extract and plot results
tibble(predictors = 1:19,
       adj_R2 = results$adjr2,
       Cp = results$cp,
       BIC = results$bic) %>%
  gather(statistic, value, -predictors) %>%
  ggplot(aes(predictors, value, color = statistic)) +
  geom_line(show.legend = F) +
  geom_point(show.legend = F) +
  facet_wrap(~ statistic, scales = "free")

#Here we see that our results identify slightly different models that are considered the best. The adjusted R^2 statistic suggests 
#the 10 variable model is preferred, the BIC statistic suggests the 4 variable model, and the Cp suggests the 8 variable model.

which.max(results$adjr2)
which.min(results$bic)
which.min(results$cp)


#We can compare the variables and coefficients that these models include using the coef function.

# 10 variable model
coef(best_subset, 10)

# 4 variable model
coef(best_subset, 4)

# 8 variable model
coef(best_subset, 8)

#We could perform the same process using forward and backward stepwise selection and obtain even more options for optimal models. 
#For example, if I assess the optimal Cp for forward and backward stepwise we see that they suggest that an 8 variable model 
#minimizes the Cp statistic, similar to the best subset approach above.

forward <- regsubsets(Salary ~ ., train, nvmax = 19, method = "forward")
backward <- regsubsets(Salary ~ ., train, nvmax = 19, method = "backward")

# which models minimize Cp?
which.min(summary(forward)$cp)
which.min(summary(backward)$cp)

#However, when we assess these models we see that the 8 variable models include different predictors. Although, all models include 
#AtBat, Hits, Walks, CWalks, and PutOuts, there are unique variables in each model.


coef(best_subset, 8)
coef(forward, 8)
coef(backward, 8)

#This highlights two important findings:

#Different subsetting procedures (best subset vs. forward stepwise vs. backward stepwise) will likely identify different "best" models.
#Different indirect error test estimate statistics (Cp, AIC, BIC, and Adjusted R^2) will likely identify different "best" models.

#This is why it is important to always perform validation; that is, to always estimate the test error directly either by using a validation 
#set or using cross-validation.

#======================================================================================
#Directly Estimating Test Error

#We now compute the validation set error for the best model of each model size. We first make a model matrix from the test data. The 
#model.matrix function is used in many regression packages for build- ing an "X" matrix from data.

test_m <- model.matrix(Salary ~ ., data = test)

#We can loop through each model size (i.e. 1 variable, 2 variables,..., 19 variables) and extract the coefficients for the best model of 
#that size, multiply them into the appropriate columns of the test model matrix to form the predictions, and compute the test MSE.

# create empty vector to fill with error values
validation_errors <- vector("double", length = 19)

for(i in 1:19) {
  coef_x <- coef(best_subset, id = i)                     # extract coefficients for model size i
  pred_x <- test_m[ , names(coef_x)] %*% coef_x           # predict salary using matrix algebra
  validation_errors[i] <- mean((test$Salary - pred_x)^2)  # compute test error btwn actual & predicted salary
}

# plot validation errors
plot(validation_errors, type = "b")

#Here, we actually see that the 1 variable model produced by the best subset approach produces the lowest test MSE. If we repeat this 
#using a different random value seed, we will get a slightly different model that is the "best". 

# this is to be expected when using a training vs. testing validation approach.

# create training - testing data
set.seed(5)
sample <- sample(c(TRUE, FALSE), nrow(hitters), replace = T, prob = c(0.6,0.4))
train <- hitters[sample, ]
test <- hitters[!sample, ]

# perform best subset selection
best_subset <- regsubsets(Salary ~ ., train, nvmax = 19)

# compute test validation errors
test_m <- model.matrix(Salary ~ ., data = test)
validation_errors <- vector("double", length = 19)

for(i in 1:19) {
  coef_x <- coef(best_subset, id = i)                     # extract coefficients for model size i
  pred_x <- test_m[ , names(coef_x)] %*% coef_x           # predict salary using matrix algebra
  validation_errors[i] <- mean((test$Salary - pred_x)^2)  # compute test error btwn actual & predicted salary
}

# plot validation errors
plot(validation_errors, type = "b")

#A more robust approach is to perform cross validation. But before we do, we should turn our our approach above for computing test errors 
#into a function. Our function pretty much mimics what we did above. The only complex part is how we extracted the formula used in the 
#call to regsubsets.

predict.regsubsets <- function(object, newdata, id ,...) {
  form <- as.formula(object$call[[2]]) 
  mat <- model.matrix(form, newdata)
  coefi <- coef(object, id = id)
  xvars <- names(coefi)
  mat[, xvars] %*% coefi
}

#We now try to choose among the models of different sizes using k-fold cross-validation. This approach is somewhat involved, as we 
#must perform best subset selection within each of the k training sets. First, we create a vector that allocates each observation to 
#one of k = 10 folds, and we create a matrix in which we will store the results.

k <- 10
set.seed(1)
folds <- sample(1:k, nrow(hitters), replace = TRUE)
cv_errors <- matrix(NA, k, 19, dimnames = list(NULL, paste(1:19)))

#Now we write a for loop that performs cross-validation. In the jth fold, the elements of folds that equal j are in the test set, 
#and the remainder are in the training set. We make our predictions for each model size, compute the test errors on the appropriate 
#subset, and store them in the appropriate slot in the matrix cv_errors.

for(j in 1:k) {
  
  # perform best subset on rows not equal to j
  best_subset <- regsubsets(Salary ~ ., hitters[folds != j, ], nvmax = 19)
  
  # perform cross-validation
  for( i in 1:19) {
    pred_x <- predict.regsubsets(best_subset, hitters[folds == j, ], id = i)
    cv_errors[j, i] <- mean((hitters$Salary[folds == j] - pred_x)^2)
  }
}

#This has given us a 10 by 19 matrix, of which the (i,j)th element corresponds to the test MSE for the ith cross-validation fold for 
#the best j-variable model. We use the colMeans function to average over the columns of this matrix in order to obtain a vector 
#for which the jth element is the cross-validation error for the j-variable model.

mean_cv_errors <- colMeans(cv_errors)

plot(mean_cv_errors, type = "b")

#We see that our more robust cross-validation approach selects an 11-variable model. We can now perform best subset selection on 
#the full data set in order to obtain the 11-variable model.

final_best <- regsubsets(Salary ~ ., data = hitters , nvmax = 19)
coef(final_best, 11)

#We see that the best 11 variable model includes variables AtBat, Hits, Walks, CAtBat, CRuns, CRBI, CWalks, LeagueN, DivisionW,
#PutOuts, and Assists.