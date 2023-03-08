## Load packages, libraries, and data:

library(mlr)
library(tidyverse)
library(mlbench)
#mlbench package#

data(Zoo)

ZooTib = as.tibble(Zoo)

ZooTib

#type variable is a factor contain- ing the animal classes we wish to predict#

# mlr doesn’t let us create tasks with logical predictors, so since all our columns are of logical class, we would have to convert them to factor#

zooTib <- mutate_all(ZooTib, as.factor)

##Training the decision tree model

# Creating the task and learner
ZooTask <- makeClassifTask(data = zooTib, target = "type")
Tree <- makeLearner("classif.rpart")

## Hyperparameter tuning

# the most important hyperparameters for tuning: minsplit, minbucket, cp, and maxdepth:

# minsplit: Minimum number of cases in a node before splitting
#  maxdepth: Maximum depth of the tree
# cp: complexity parameter: Minimum improvement in performance for a split
# minbucket: Minimum number of cases in a leaf

# few others: 
# maxcompete: hyperparameter controls how many candidate splits can be displayed for each node in the model summary. But tuning maxcompete doesn’t affect model performance, only its summary.
# maxsurrogate: controls how many surrogate splits are shown. A surrogate split is a split used if a particular case is missing data for the actual split. In this way, rpart can handle missing data as it learns which splits can be used in place of missing variables. We can quickly count the number of missing values per column of a data.frame or tibble by running map_dbl(zooTib, ~sum(is.na(.))).
# usesurrogate: hyperparameter controls how the algorithm uses surrogate splits. A value of zero means surrogates will not be used, and cases with missing data will not be classified.

# Printing available rpart hyperparameters

getParamSet(Tree)
# Defining the hyperparameter space for tuning
# Remember that we use makeIntegerParam() and makeNumericParam() to define the search spaces for integer and numeric hyperparameters, respec- tively.

TreeParamSpace <- makeParamSet(
  makeIntegerParam("minsplit", lower = 5, upper = 30),
  makeIntegerParam("minbucket", lower = 3, upper = 10),
  makeNumericParam("cp", lower = 0.01, upper = 0.1),
  makeIntegerParam("maxdepth", lower = 3, upper = 10))

# Next, we can define how we’re going to search the hyperparameter space we defined. Because the hyperparameter space is quite large, we’re going to use a random search rather than a grid search.

# Also, we also define our cross-validation strategy for tuning. Here, we are going to use ordinary 5-fold cross-validation.

# Note: Ordinarily, if classes are imbalanced, I would use stratified sampling. Here, though, because we have very few cases in some of the classes, there are not enough cases to stratify (try it: you’ll get an error). For this example, we won’t stratify; but in situations where you have very few cases in a class, you should consider whether there is enough data to justify keeping that class in the model.

RandSearch <- makeTuneControlRandom(maxit = 100)
CvForTuning <- makeResampleDesc("CV", iters = 5)

## Performing hyperparameter tuning in parellel

library(parallel)
library(parallelMap)

parallelStartSocket(cpus = detectCores())

tunedTreePars <- tuneParams(Tree, task = ZooTask,
                            resampling = CvForTuning,
                            par.set = TreeParamSpace,
                            control = RandSearch)

parallelStop()

tunedTreePars

# Note: The rpart algorithm isn’t nearly as computationally expensive as the support vector machine (SVM) algorithm.

# Now that we’ve tuned our hyperparameters, we can train our final model using them. We use the setHyperPars() function to create a learner using the tuned hyperparameters, which we access using tunedTreePars$x. We can then train the final model using the train() function, as usual.

TunedTree <- setHyperPars(Tree, par.vals = tunedTreePars$x)
TunedTreeModel <- train(TunedTree, ZooTask)
# The easiest way to interpret the model is to draw a graphical representation of the tree. There are a few ways of plotting decision tree models in R, but my favorite is the rpart.plot() function from the package of the same name. Let’s install the rpart.plot package first and then extract the model data using the getLearnerModel() function.

install.packages("rpart.plot")
library(rpart.plot)
treeModelData <- getLearnerModel(TunedTreeModel)
rpart.plot(treeModelData, roundint = FALSE,
           box.palette = "BuBn",
           type = 5)

# Exploring the model

printcp(treeModelData, digits = 3)

# For a detailed summary of the model, run summary(treeModelData)

## Cross-validating our decision tree model

# First, we define our outer cross-validation strategy. This time I’m using 5-fold cross- validation as my outer cross-validation loop. We’ll use the cvForTuning resampling description we made in listing 7.6 for the inner loop.
# Next, we create our wrapper by “wrapping together” our learner and hyperparam- eter tuning process. We supply our inner cross-validation strategy, hyperparameter space, and search method to the makeTuneWrapper() function.
# Finally, we can start parallelization with the parallelStartSocket() function, and start the cross-validation process with the resample() function. The resample() func- tion takes our wrapped learner, task, and outer cross-validation strategy as arguments.

outer <- makeResampleDesc("CV", iters = 5)
treeWrapper <- makeTuneWrapper("classif.rpart", resampling = CvForTuning,
                               par.set = TreeParamSpace,
                               control = RandSearch)
parallelStartSocket(cpus = detectCores())

cvWithTuning <- resample(treeWrapper, ZooTask, resampling = outer)

parallelStop()

## Extracting the cross-validation result

cvWithTuning

# If for example: During hyperparameter tuning, the best hyperparameter combination gave us a mean misclassification error (MMCE) of 0.0698. But our cross-validated estimate of model performance gives us an MMCE of 0.12. Quite a large difference! What’s going on? Well, this is an example of overfitting. Our model is performing better during hyper- parameter tuning than during cross-validation. This is also a good example of why it’s important to include hyperparameter tuning inside our cross-validation procedure.
# We’ve just discovered the main problem with the rpart algorithm (and decision trees in general): they tend to produce models that are overfit. How do we overcome this problem? The answer is to use an ensemble method, an approach where we use mul- tiple models to make predictions for a single task.







