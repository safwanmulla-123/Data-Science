#Improving decision Tree with random forests and boosting

##Ensemble techniques: Bagging, boosting, and stacking

###Ensemble methods, such as random forest and gradient boosting, which combine multiple trees to make predictions.

###Benchmarking is the process of letting a bunch of different learning algorithms fight it out to select the one that performs best for a particular problem.

###Importing the code from last chapter
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

#install.packages("rpart.plot")
library(rpart.plot)
treeModelData <- getLearnerModel(TunedTreeModel)
#rpart.plot(treeModelData, roundint = FALSE,
#           box.palette = "BuBn",
#           type = 5)

# Exploring the model

printcp(treeModelData, digits = 3)

#Starting Chapter 3-

###There are three different ensemble methods:
#### - Bootstrap aggregating
#### - Boosting
#### - Stacking

##Training models on sampled data: Bootstrap aggregating

###Bootstrap aggregating (a.k.a. - bagging) A technique used to be able to use all the available data, while looking past the noisy data and reduce prediction variance.

###Premise of bagging: 
#### - Number of sub-models to train
#### - For each sub-model, randomly sample cases from the training set with replacements until we have a sample the same size as the original training set
#### - Train a sub-model on each sample of cases.
#### - Pass new data through each sub-model, and let them vote on the prediction.
#### - The modal prediction (the most frequent prediction) from all the sub-models is used as the predicted output.

####The random forest algorithm uses bagging to create a large number of trees. These trees are saved as part of the model; when we pass the model new data, each tree makes its own prediction, and the modal prediction is returned.

## Boosting method:
#### Learning from the previous models’ mistakes.
#### Boosting is used in algorithms called AdaBoost, XGBoost, and others. With bagging, the individ- ual models are trained in parallel. In contrast, boosting is an ensemble technique that, again, trains many individual models, but builds them sequentially.
#### Just like bagging, boosting can be applied to any supervised machine learning algorithm. However, boosting is most beneficial when using weak learners as the sub- models. By weak learner, I don’t mean someone who keeps failing their driving test; I mean a model that only does a little better at making predictions than a random guess. For this reason, boosting has been traditionally applied to shallow decision trees. By shallow, I mean a decision tree that doesn’t have many levels of depth, or may have only a single split.
#### The function of boosting is to combine many weak learners together to form one strong ensemble learner. The reason we use weak learners is that there is no improvement in model performance when boosting with strong learners versus weak learners. So why waste computational resources training hundreds of strong, probably more complex learners, when we can get the same performance by training weak, less complex ones?

### Methods of boosting:
#### - Adaptive boosting
#### - Gradient boosting

### Adaptive Boosting:
#### - WEIGHTING INCORRECTLY PREDICTED CASES
#### - Well known adaptive boosting algorithm -  AdaBoost
#### - Initially, all cases in the training set have the same importance, or weight. An initial model is trained on a boot- strap sample of the training set where the probability of a case being sampled is pro- portional to its weight (all equal at this point). The cases that this initial model incorrectly classifies are given more weight/importance, while cases that it correctly classifies are given less weight/importance.
#### - The next model takes another bootstrap sample from the training set, but the weights are no longer equal. This ensures that cases incorrectly classified by the previous model are more likely to be featured in the bootstrap for the subsequent model. The subsequent model is therefore more likely to learn rules that will correctly classify these cases.
#### - This process continues: a new model is added to the ensemble, all the models vote, weights are updated, and the next model samples the data based on the new weights. Once we reach the maximum number of predefined trees, the process stops, and we get our final ensemble model. 
#### - When unseen cases are passed to the final model for prediction, each tree votes individually (like in bagging), but each vote is weighted by the model weight.

### Gradient Boosting:
#### - Learning from the residuals of previous model
#### - Rather than weighting the cases differently depending on the accuracy of their classification, subsequent models try to predict the residuals of the previous ensemble of models.
#### - A residual, or residual error, is the difference between the true value (the “observed” value) and the value predicted by a model.
#### - we can quantify the residual error of a classification model as: a) The proportion of all cases incorrectly classified. b) The log loss
#### - The proportion of cases that were misclassified is pretty self-explanatory. 
#### - The log loss is similar but more greatly penalizes a model that makes incorrect classifications confi- dently. If your friend tells you with “absolute certainty” that Helsinki is the capital of Sweden (it’s not), you’d think less of them than if they said they “think it might be” the capital. This is how log loss treats misclassification error.
#### - Sampling in stochastic gradient descent is usually without replacement, which means it isn’t a bootstrap sample.

#### - the best known gradient boosting algorithm is the XGBoost (extreme gradient boosting) algorithm.

### XGBoost:
#### It can build different branches of each tree in parallel, speeding up model building.
#### It can handle missing data.
#### It employs regularization. 

## Stacking Method:
#### Learning from predictions made by other models
#### In bagging and boosting, the learners are often (but don’t always have to be) homo- geneous. 
#### Put another way, all of the sub-models were learned by the same algorithm (decision trees). Stacking explicitly uses different algorithms to learn the sub-models. 
#### For example, we may choose to use the kNN algorithm, logistic regression algorithm, and the SVM algorithm to build three independent base models.

### Note: Ensemble methods like bagging, boosting, and stacking are not strictly machine learning algorithms in their own right. They are algorithms that can be applied to other machine learning algorithms. Ensembling is most commonly applied to tree-based learners; but we could just as easily apply bagging and boosting to other machine learning algorithms, such as kNN and linear regression.

## Random Forest Model:

### Making Learner for Random Forest:

forest <- makeLearner("classif.randomForest")

### Tuning the random forest hyperparameters:

#### Creating hyperparameter tuning space-

forestParamSpace <- makeParamSet(
  makeIntegerParam("ntree", lower = 300, upper = 300),
  makeIntegerParam("mtry", lower = 6, upper = 12),
  makeIntegerParam("nodesize", lower = 1, upper = 5),
  makeIntegerParam("maxnodes", lower = 5, upper = 20)
)

#### Define a random search method with 100 iterations-

randSearch <- makeTuneControlRandom(maxit = 100)

#### Define a 5-fold CV strategy-

cvForTuning <- makeResampleDesc("CV", iter = 5)

parallelStartSocket(cpus = detectCores())

#### Tuning the hyperparameters"

tunedForestPars <- tuneParams(forest, task = ZooTask,
                              resampling = cvForTuning,
                              par.set = forestParamSpace,
                              control = randSearch)
parallelStop()

#### Printing results-

tunedForestPars

### Training the final model:

tunedForest <- setHyperPars(forest, par.vals = tunedForestPars$x)

tunedForestModel <- train(tunedForest, ZooTask)

### Plotting the out-of-bag error:
####(The out-of-bag error is the mean prediction error for each case, by trees that did not include that case in their bootstrap. Out-of-bag error estimation is specific to algorithms that use bagging and allows us to estimate the performance of the forest as it grows.)

forestModelData <- getLearnerModel(tunedForestModel)

species <- colnames(forestModelData$err.rate)

plot(forestModelData, col = 1:length(species), lty = 1:length(species))

legend("topright", species,
       col = 1:length(species),
       lty = 1:length(species))

### Cross validating the model-building process:
outer <- makeResampleDesc("CV", iters = 5)

forestWrapper <- makeTuneWrapper("classif.randomForest",
                                 resampling = cvForTuning,
                                 par.set = forestParamSpace,
                                 control = randSearch)

parallelStartSocket(cpus = detectCores())

cvWithTuning <- resample(forestWrapper, ZooTask, resampling = outer)

parallelStop()

cvWithTuning

## Building XGBoost model:

xgb <- makeLearner("classif.xgboost")

zooXgb <- mutate_at(zooTib, .vars = vars(-type),
                    .funs = as.numeric)

xgbTask <- makeClassifTask(data = zooXgb, target = "type")

xgbParamSpace <- makeParamSet(
  makeNumericParam("eta", lower = 0, upper = 1),
  makeNumericParam("gamma", lower = 0, upper = 5),
  makeIntegerParam("max_depth", lower = 1, upper = 5),
  makeNumericParam("min_child_weight", lower = 1, upper = 10),
  makeNumericParam("subsample", lower = 0.5, upper = 1),
  makeNumericParam("colsample_bytree", lower = 0.5, upper = 1),
  makeIntegerParam("nrounds", lower = 20, upper = 20),
  makeDiscreteParam("eval_metric", values = c("merror", "mlogloss")))

randSearch <- makeTuneControlRandom(maxit = 1000)

cvForTuning <- makeResampleDesc("CV", iters = 5)

tunedXgbPars <- tuneParams(xgb, task = xgbTask,
                           resampling = cvForTuning,
                           par.set = xgbParamSpace,
                           control = randSearch)

tunedXgbPars

### Training the final tuned model:

tunedXgb <- setHyperPars(xgb, par.vals = tunedXgbPars$x)

tunedXgbModel <- train(tunedXgb, xgbTask)

### Plotting iteration number against log loss:

xgbModelData <- getLearnerModel(tunedXgbModel)

ggplot(xgbModelData$evaluation_log, aes(x = iter, y = train_merror)) +
  geom_line() +
  geom_point()

xgboost::xgb.plot.tree(model = xgbModelData, trees = 1:5)

### Plotting individual decision tree:

outer <- makeResampleDesc("CV", iters = 3)
xgbWrapper <- makeTuneWrapper("classif.xgboost",
                              resampling = cvForTuning,
                              par.set = xgbParamSpace,
                              control = randSearch)

cvWithTuning <- resample(xgbWrapper, xgbTask, resampling = outer)

cvWithTuning


## Benchmarking algorithms against each other:

learners = list(makeLearner("classif.knn"),
                makeLearner("classif.svm"),
     #           tunedTree,
                tunedForest,
                tunedXgb)
benchCV <- makeResampleDesc("RepCV", folds = 10, reps = 5)
bench <- benchmark(learners, xgbTask, benchCV)

bench






































