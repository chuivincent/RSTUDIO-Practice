# Regression Trees

#Basic regression trees partition a data set into smaller groups and then fit a simple 
#model (constant) for each subgroup. Unfortunately, a single tree model tends to be 
#highly unstable and a poor predictor. However, by bootstrap aggregating (bagging) 
#regression trees, this technique can become quite powerful and effective. Moreover, 
#this provides the fundamental basis of more complex tree-based models such as random
#forests and gradient boosting machines.


install.packages('rsample')
install.packages('dplyr')
install.packages('rpart')
install.packages('rpart.plot')
install.packages('ipred')
install.packages('caret')
install.packages('AmesHousing')
library(rsample)     # data splitting 
library(dplyr)       # data wrangling
library(rpart)       # performing regression trees
library(rpart.plot)  # plotting regression trees
library(ipred)       # bagging
library(caret)       # bagging
library(AmesHousing) # Housing Data

# Create training (70%) and test (30%) sets for the AmesHousing::make_ames() data.
# Use set.seed for reproducibility

set.seed(123)
ames_split <- initial_split(AmesHousing::make_ames(), prop = .7)
ames_train <- training(ames_split)
ames_test  <- testing(ames_split)

#There are many methodologies for constructing regression trees but one of the oldest is 
#known as the classification and regression tree (CART) approach developed by Breiman et 
#al. (1984) Basic regression trees partition a data set into smaller subgroups and then 
#fit a simple constant for each observation in the subgroup. The partitioning is achieved 
#by successive binary partitions (aka recursive partitioning) based on the different 
#predictors. The constant to predict is based on the average response values for all 
#observations that fall in that subgroup.

#For example, consider we want to predict the miles per gallon a car will average based 
#on cylinders (cyl) and horsepower (hp). All observations go through this tree, are 
#assessed at a particular node, and proceed to the left if the answer is yes or proceed 
#to the right if the answer is no. So, first, all observations that have 6 or 8 cylinders
#go to the left branch, all other observations proceed to the right branch. Next, the left
#branch is further partitioned by horsepower. Those 6 or 8 cylinder observations with horsepower
#equal to or greater than 192 proceed to the left branch; those with less than 192 hp 
#proceed to the right. These branches lead to terminal nodes or leafs which contain our 
#predicted response value. Basically, all observations (cars in this example) that do not
#have 6 or 8 cylinders (far right branch) average 27 mpg. All observations that have 6 or
#8 cylinders and have more than 192 hp (far left branch) average 13 mpg.


#This simple example can be generalized to state we have a continuous response variable Y
#and two inputs X1 and X2. The recursive partitioning results in three regions (R1,R2,R3) 
#where the model predicts Y with a constant $c_m$ for region Rm

#Deciding on splits
#First, its important to realize the partitioning of variables are done in a top-down, 
#greedy fashion. This just means that a partition performed earlier in the tree will not 
#change based on later partitions. But how are these partions made? The model begins with 
#the entire data set, S, and searches every distinct value of every input variable to find
#the predictor and split value that partitions the data into two regions (R1 and R2) such 
#that the overall sums of squares error are minimized

#Having found the best split, we partition the data into the two resulting regions and 
#repeat the splitting process on each of the two regions. This process is continued until 
#some stopping criterion is reached. What results is, typically, a very deep, complex 
#tree that may produce good predictions on the training set, but is likely to overfit the
#data, leading to poor performance on unseen data.

#Cost complexity criterion
#There is often a balance to be achieved in the depth and complexity of the tree to optimize
#predictive performance on some unseen data. To find this balance, we typically grow a 
#very large tree as defined in the previous section and then prune it back to find an 
#optimal subtree.We find the optimal subtree by using a cost complexity parameter (α) 
#that penalizes our objective function

#For a given value of α, we find the smallest pruned tree that has the lowest penalized 
#error. If you are familiar with regularized regression, you will realize the close 
#association to the lasso L2 norm penalty. As with these regularization methods, smaller 
#penalties tend to produce more complex models, which result in larger trees. Whereas larger
#penalties result in much smaller trees. Consequently, as a tree grows larger, the 
#reduction in the SSE must be greater than the cost complexity penalty. Typically, we 
#evaluate multiple models across a spectrum of α and use cross-validation to identify the 
#optimal α and, therefore, the optimal subtree.

#Strengths and weaknesses
#There are several advantages to regression trees:
  
#They are very interpretable.

#Making predictions is fast (no complicated calculations, just looking up constants in 
#the tree).

#It is easy to understand what variables are important in making the prediction. The 
#internal nodes (splits) are those variables that most largely reduced the SSE.

#If some data is missing, we might not be able to go all the way down the tree to a leaf,
#but we can still make a prediction by averaging all the leaves in the sub-tree we do 
#reach.

#The model provides a non-linear jagged response, so it can work when the true 
#regression surface is not smooth. If it is smooth, though, the piecewise-constant 
#surface can approximate it arbitrarily closely (with enough leaves).

#There are fast, reliable algorithms to learn these trees.

#But there are also some significant weaknesses:
  
#Single regression trees have high variance, resulting in unstable predictions 
#(an alternative subsample of training data can significantly change the terminal nodes).

#Due to the high variance single regression trees have poor predictive accuracy.

#Basic Implementation
#We can fit a regression tree using rpart and then visualize it using rpart.plot. The 
#fitting process and the visual output of regression trees and classification trees 
#are very similar. Both use the formula method for expressing the model (similar to lm). 
#However, when fitting a regression tree, we need to set method = "anova". By default, 
#rpart will make an intelligent guess as to what the method value should be based on the
#data type of your response column, but it is recommened that you explictly set the method
#for reproducibility reasons (since the auto-guesser may change in the future).

m1 <- rpart(
  formula = Sale_Price ~ .,
  data    = ames_train,
  method  = "anova"
)

#Once we fit our model we can take a peak at the m1 output. This just explains steps 
#of the splits. For example, we start with 2051 observations at the root node (very 
#beginning) and the first variable we split on (the first variable that optimizes a 
#reduction in SSE) is Overall_Qual. We see that at the first node all observations with 
#Overall_Qual=Very_Poor,Poor,Fair,Below_Average,Average,Above_Average,Good go to the 2nd 
#(2)) branch. The total number of observations that follow this branch (1699), their 
#average sales price (156147.10) and SSE (4.001092e+12) are listed. If you look for the 
#3rd branch (3)) you will see that 352 observations with Overall_Qual=Very_Good,Excellent,
#Very_Excellent follow this branch and their average sales prices is 304571.10 and the 
#SEE in this region is 2.874510e+12. Basically, this is telling us the most important 
#variable that has the largest reduction in SEE initially is Overall_Qual with those 
#homes on the upper end of the quality spectrum having almost double the average sales
#price.

m1

#We can visualize our model with rpart.plot. rpart.plot has many plotting options. 
#In the default print it will show the percentage of data that fall to that node and the 
#average sales price for that branch. 
#One thing you may notice is that this tree contains 11 internal nodes resulting in 12 
#terminal nodes. Basically, this tree is partitioning on 11 variables to produce its model. 
#However, there are 80 variables in ames_train. 

rpart.plot(m1)

#Behind the scenes rpart is automatically applying a range of cost complexity (α)
#values to prune the tree. To compare the error for each α value, rpart performs a 
#10-fold cross validation so that the error associated with a given α value is 
#computed on the hold-out validation data. In this example we find diminishing 
#returns after 12 terminal nodes as illustrated below (y-axis is cross validation 
#error, lower x-axis is cost complexity (α) value, upper x-axis is the number of
#terminal nodes (tree size = |T|). You may also notice the dashed line which goes 
#through the point |T|=9. Breiman et al. (1984) suggested that in actual practice, 
#its common to instead use the smallest tree within 1 standard deviation of the 
#minimum cross validation error (aka the 1-SE rule). Thus, we could use a tree with
#9 terminal nodes and reasonably expect to experience similar results within a small 
#margin of error.

plotcp(m1)

#To illustrate the point of selecting a tree with 12 terminal nodes (or 9 if you go 
#by the 1-SE rule), we can force rpart to generate a full tree by using cp = 0 
#(no penalty results in a fully grown tree). We can see that after 12 terminal nodes, 
#we see diminishing returns in error reduction as the tree grows deeper. Thus, we 
#can signifcantly prune our tree and still achieve minimal expected error.

m2 <- rpart(
  formula = Sale_Price ~ .,
  data    = ames_train,
  method  = "anova", 
  control = list(cp = 0, xval = 10)
)

m2
rpart.plot(m2)
plotcp(m2)
abline(v = 12, lty = "dashed")


#So, by default, rpart is performing some automated tuning, with an optimal subtree 
#of 11 splits, 12 terminal nodes, and a cross-validated error of 0.272 (note that 
#this error is equivalent to the PRESS statistic but not the MSE). However, we can
#perform additional tuning to try improve model performance.

m1$cptable

#Tuning

#In addition to the cost complexity (α) parameter, it is also common to tune:
  
#minsplit: the minimum number of data points required to attempt a split before it 
#is forced to create a terminal node. The default is 20. Making this smaller allows 
#for terminal nodes that may contain only a handful of observations to create the 
#predicted value.

#maxdepth: the maximum number of internal nodes between the root node and the 
#terminal nodes. The default is 30, which is quite liberal and allows for fairly 
#large trees to be built.

#rpart uses a special control argument where we provide a list of hyperparameter 
#values. For example, if we wanted to assess a model with minsplit = 10 and 
#maxdepth = 12, we could execute the following

m3 <- rpart(
  formula = Sale_Price ~ .,
  data    = ames_train,
  method  = "anova", 
  control = list(minsplit = 10, maxdepth = 12, xval = 10)
)

m3

rpart.plot(m3)

plotcp(m3)
abline(v = 10, lty = "dashed")

m3$cptable


#Although useful, this approach requires you to manually assess multiple models. 
#Rather, we can perform a grid search to automatically search across a range of 
#differently tuned models to identify the optimal hyerparameter setting.

#To perform a grid search we first create our hyperparameter grid. In this example, 
#I search a range of minsplit from 5-20 and vary maxdepth from 8-15 (since our 
#original model found an optimal depth of 12). What results is 128 different 
#combinations, which requires 128 different models.

hyper_grid <- expand.grid(
  minsplit = seq(5, 20, 1),
  maxdepth = seq(8, 15, 1)
)

head(hyper_grid)

nrow(hyper_grid)

#To automate the modeling we simply set up a for loop and iterate through each 
#minsplit and maxdepth combination. We save each model into its own list item.

models <- list()

for (i in 1:nrow(hyper_grid)) {
  
  # get minsplit, maxdepth values at row i
  minsplit <- hyper_grid$minsplit[i]
  maxdepth <- hyper_grid$maxdepth[i]
  
  # train a model and store in the list
  models[[i]] <- rpart(
    formula = Sale_Price ~ .,
    data    = ames_train,
    method  = "anova",
    control = list(minsplit = minsplit, maxdepth = maxdepth)
  )
}

#We can now create a function to extract the minimum error associated with the 
#optimal cost complexity α value for each model. After a little data wrangling to 
#extract the optimal α value and its respective error, adding it back to our grid, 
#and filter for the top 5 minimal error values we see that the optimal model makes a
#slight improvement over our earlier model (xerror of 0.242 versus 0.272).

# function to get optimal cp
get_cp <- function(x) {
  min    <- which.min(x$cptable[, "xerror"])
  cp <- x$cptable[min, "CP"] 
}

# function to get minimum error
get_min_error <- function(x) {
  min    <- which.min(x$cptable[, "xerror"])
  xerror <- x$cptable[min, "xerror"] 
}

hyper_grid %>%
  mutate(
    cp    = purrr::map_dbl(models, get_cp),
    error = purrr::map_dbl(models, get_min_error)
  ) %>%
  arrange(error) %>%
  top_n(-5, wt = error)

#If we were satisfied with these results we could apply this final optimal model 
#and predict on our test set. The final RMSE is 39145.39 which suggests that, on 
#average, our predicted sales prices are about $39,145 off from the actual sales 
#price.

optimal_tree <- rpart(
  formula = Sale_Price ~ .,
  data    = ames_train,
  method  = "anova",
  control = list(minsplit = 15, maxdepth = 12, cp = 0.01)
)

pred <- predict(optimal_tree, newdata = ames_test)
RMSE(pred = pred, obs = ames_test$Sale_Price)

#Bagging

#The idea
#As previously mentioned, single tree models suffer from high variance. Although 
#pruning the tree helps reduce this variance, there are alternative methods that 
#actually exploite the variability of single trees in a way that can significantly 
#improve performance over and above that of single trees. Bootstrap aggregating 
#(bagging) is one such approach (originally proposed by Breiman, 1996).

#Bagging combines and averages multiple models. Averaging across multiple trees 
#reduces the variability of any one tree and reduces overfitting, which improves 
#predictive performance. Bagging follows three simple steps:

#Create m bootstrap samples from the training data. Bootstrapped samples allow us 
#to create many slightly different data sets but with the same distribution as the 
#overall training set.

#For each bootstrap sample train a single, unpruned regression tree.

#Average individual predictions from each tree to create an overall average predicted value.

#This process can actually be applied to any regression or classification model; however, 
#it provides the greatest improvement for models that have high variance. For example, 
#more stable parametric models such as linear regression and multi-adaptive regression 
#splines tend to experience less improvement in predictive performance.


#One benefit of bagging is that, on average, a bootstrap sample will contain 63% of the 
#training data. This leaves about 33% of the data out of the bootstrapped sample. We call 
#this the out-of-bag (OOB) sample. We can use the OOB observations to estimate the model’s 
#accuracy, creating a natural cross-validation process.

#Bagging with ipred
#Fitting a bagged tree model is quite simple. Instead of using rpart we use ipred::bagging.
#We use coob = TRUE to use the OOB sample to estimate the test error. We see that our initial 
#estimate error is close to $3K less than the test error we achieved with our single optimal 
#tree (36543 vs. 39145)

# make bootstrapping reproducible
set.seed(123)

# train bagged model
bagged_m1 <- bagging(
  formula = Sale_Price ~ .,
  data    = ames_train,
  coob    = TRUE
)

bagged_m1

#One thing to note is that typically, the more trees the better. As we add more trees we are 
#averaging over more high variance single trees. What results is that early on, we see a 
#dramatic reduction in variance (and hence our error) and eventually the reduction in error 
#will flatline signaling an appropriate number of trees to create a stable model. Rarely will
#you need more than 50 trees to stabilize the error.

#By default bagging performs 25 bootstrap samples and trees but we may require more. We can assess 
#the error versus number of trees as below. We see that the error is stabilizing at about 25 trees 
#so we will likely not gain much improvement by simply bagging more trees.


# assess 10-50 bagged trees
ntree <- 10:50

# create empty vector to store OOB RMSE values
rmse <- vector(mode = "numeric", length = length(ntree))

for (i in seq_along(ntree)) {
  # reproducibility
  set.seed(123)
  
  # perform bagged model
  model <- bagging(
    formula = Sale_Price ~ .,
    data    = ames_train,
    coob    = TRUE,
    nbagg   = ntree[i]
  )
  # get OOB error
  rmse[i] <- model$err
}

plot(ntree, rmse, type = 'l', lwd = 2)
abline(v = 25, col = "red", lty = "dashed")

#Bagging with caret
#Bagging with ipred is quite simple; however, there are some additional benefits of bagging with caret.

#Its easier to perform cross-validation. Although we can use the OOB error, performing cross validation 
#will also provide a more robust understanding of the true expected test error. We can assess variable 
#importance across the bagged trees.

#Here, we perform a 10-fold cross-validated model. We see that the cross-validated RMSE is $36,477. We 
#also assess the top 20 variables from our model. Variable importance for regression trees is measured 
#by assessing the total amount SSE is decreased by splits over a given predictor, averaged over all m trees. 
#The predictors with the largest average impact to SSE are considered most important. The importance value 
#is simply the relative mean decrease in SSE compared to the most important variable (provides a 0-100 scale).

# Specify 10-fold cross validation
ctrl <- trainControl(method = "cv",  number = 10) 

# CV bagged model
bagged_cv <- train(
  Sale_Price ~ .,
  data = ames_train,
  method = "treebag",
  trControl = ctrl,
  importance = TRUE
)

# assess results
bagged_cv

# plot most important variables
plot(varImp(bagged_cv), 20)  

#If we compare this to the test set out of sample we see that our cross-validated error estimate was very close. 
#We have successfully reduced our error to about $35,000;extensions of this bagging concept (random forests and GBMs) 
#can significantly reduce this further.

pred <- predict(bagged_cv, ames_test)
RMSE(pred, ames_test$Sale_Price)
