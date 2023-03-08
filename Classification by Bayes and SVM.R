#Bayes Algorithm:

#### Discriminant analysis algorithms use Bayes’ rule to pre- dict the probability of a case belonging to each of the classes, based on its discrimi- nant function values. The naive Bayes algorithm works in exactly the same way, except that it doesn’t perform dimension reduction as discriminant analysis does, and it can handle categorical, as well as continuous, predictors.

#### Bayes Rule: p(k|x) = [p(x|k) x p(k)] / p(x)

####Where:
###What is the naive Bayes algorithm? 137
# p(k|x) is the probability of having the disease (k) given a positive test result (x). This is called the posterior probability.
# p(x|k) is the probability of getting a positive test result if you do have the disease. This is called the likelihood.
# p(k) is the probability of having the disease regardless of any test. This is the proportion of people in the population with the disease and is called the prior probability.
# p(x) is the probability of getting a positive test result and includes the true posi- tives and false positives. This is called the evidence.
#We can rewrite this in plain English:
#posterior = likelihood × prior / evidence

#loading necessary libraries:
library(mlr)
library(tidyverse)

#Importing Data:

data(HouseVotes84, package = "mlbench")

votesTib <- as_tibble(HouseVotes84)

votesTib
#to look into data run: ?mlbench::HouseVotes84

#Summarizing the number of missing values in each avriable using map_dbl().
####map_dbl() iterates a function over every element of a vector/list (or, in this case, every column of a tibble), applies a function to that element, and returns a vec- tor containing the function output.
####Our function passes each vector to sum(is.na(.)) to count the number of missing values in that vector. This function is applied to each column of the tibble and returns the number of missing values for each.

map_dbl(votesTib, ~sum(is.na(.)))

####The naive Bayes algorithm can handle missing data in two ways:
#### By omitting the variables with missing values for a particular case, but still using that case to train the model
#### By omitting that case entirely from the training set
####By default, the naive Bayes implementation that mlr uses is to keep cases and drop variables.

#Plotting the data to get better understanding:

votesUntidy <- gather(votesTib, "Variable", "Value", -Class)

ggplot(votesUntidy, aes(Class, fill = Value))  +
  facet_wrap(~ Variable, scales = "free_y")  +
  geom_bar(position = "fill") +
  theme_bw()

#Training the Model

####Creating task and learner, and build our model

votesTask <- makeClassifTask(data = votesTib, target = "Class")

bayes <- makeLearner("classif.naiveBayes")

bayesModel <- train(bayes, votesTask)

####Notes: Model training complete without any errors because bayes can handle missing data

####Using 10-fold CV repeated 50 times to evaluate performance of our model

kFold <- makeResampleDesc(method = "RepCV", folds = 10, reps = 50,
                          stratify = TRUE)

bayesCV <- resample(learner = bayes, task = votesTask,
                    resampling = kFold,
                    measures = list(mmce, acc, fpr, fnr))
bayesCV$aggr

####Model correctly predicts 90% of test set cases in our CV. Thats goood.

##Testing the model to make predictions:

politician <- tibble(V1 = "n", V2 = "n", V3 = "y", V4 = "n", V5 = "n",
                     V6 = "y", V7 = "y", V8 = "y", V9 = "y", V10 = "y",
                     V11 = "n", V12 = "y", V13 = "n", V14 = "n",
                     V15 = "y", V16 = "n")
politicianPred <- predict(bayesModel, newdata = politician)

getPredictionResponse(politicianPred)

#####Our model predicts that the new politician is a Democrat.

#SVM Algorithm:

### Loading the liabraries:
library(mlr)
library(tidyverse)

### Loading data:

data(spam, package = "kernlab")
spamTib <- as_tibble(spam)
spamTib

### Defining task and learner for SVM:

spamTask <- makeClassifTask(data = spamTib, target = "type")
svm <- makeLearner("classif.svm")

### Checking which hyperparameters are availabale:

getParamSet("classif.svm")

### Most important parameters to tune: Kernel, Cost, Degree, Gamma.

### First we define a vector of kernel functions we wish to tune, then we use makeParamSet() to define the hyperparameter space we which to tune over.
###Kernel functions to choose from: linear kernel (equivalent to no kernel), Polynomial kernel, Gaussian radial basis kernel, Sigmoid kernel

kernels <- c("polynomial", "radical", "sigmoid")

svmParamSpace <- makeParamSet(
  makeDiscreteParam("kernel", values = kernels),
  makeIntegerParam("degree", lower = 1, upper = 3),
  makeNumericParam("cost", lower = 0.1, upper = 10),
  makeNumericParam("gamma", lower = 0.1, 10)
)

### Instead of grid search, which tryies every possible combination of parameters requiring more time and resources, we are going to use Random search using makeTuneControlRandom() and do mulitple itterations using maxit().
### Also, since the process is computationally expensive, we use hold-out Cv instead of k-fold.

randSearch <- makeTuneControlRandom(maxit = 20)

cvForTuning <- makeResampleDesc("Holdout", split = 2/3)

tunedSVMPars <- tuneParams("classif.svm", task = spamTask,
                           resampling = cvForTuning,
                           par.set = svmParamSpace,
                           control = randSearch)

tunedSVMPars

### The results show following tuned parameters : kernel=polynomial; degree=1; cost=5.23; gamma=3 : mmce.test.mean=0.0645372

## Training the model with tuned hyperparamaters:

tunedSVM <- setHyperPars(makeLearner("classif.svm"), par.vals = tunedSVMPars$x)

tunedSVMModel <- train(tunedSVM, spamTask)

## Cross Validating the model-building process:

outer <- makeResampleDesc("CV", iters = 3)

svmWrapper <- makeTuneWrapper("classif.svm", resampling = cvForTuning,
                              par.set = svmParamSpace,
                              control = randSearch)

cvWithTuning <- resample(svmWrapper, spam.task, resampling = outer)

cvWithTuning

###We have correctly classified about 91% emails as spam or not spam.


















