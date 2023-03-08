library(mlr)
library(tidyverse)
library(mclust)

data("iris")
newData <- as_tibble(iris)

ggplot(newData, aes(x = Sepal.Length, y = Sepal.Width,
       col = Species, shape = Species)) +
  geom_point() +
  theme_bw()

ggplot(newData, aes(x = Petal.Length, y = Sepal.Length,
                    col = Species, shape = Species)) +
  geom_point() +
  theme_bw()

#creating a task:

newDataTask <- makeClassifTask(data = newData, target = "Species")

newDataTask

#Creating a learner:

kNN <- makeLearner("classif.knn", par.vals = list("k" = 2))

#Creating holdout CV:

holdoutNew <- makeResampleDesc(method = "Holdout", split = 2/3,
                               stratify = TRUE)

holdoutNewCV <- resample(learner = kNN, task = newDataTask, 
                         resampling = holdoutNew,
                         measures = list(mmce, acc))

calculateConfusionMatrix(holdoutNewCV$pred, relative = TRUE)

#Tuning k for kNN:

ParamSpacekNN <- makeParamSet(makeDiscreteParam("k", values = 1:10))

gridSearch <- makeTuneControlGrid()

cvForTuning <- makeResampleDesc(method = "Holdout", split = 2/3, 
                                stratify = TRUE)

tunedK <- tuneParams(learner = "classif.knn", task = newDataTask, 
                     resampling = cvForTuning, par.set = ParamSpacekNN,
                     control = gridSearch)
tunedK

tunedK$x

knnTuningData <- generateHyperParsEffectData(tunedK)

plotHyperParsEffect(knnTuningData, x = "k", y = "mmce.test.mean",
                    plot.type = "line") +
  theme_bw()

###Training the final model, using the tuned k value:

tunedKnn <- setHyperPars(makeLearner("classif.knn"),
                         par.vals = tunedK$x)

tunedKnnModel <- train(tunedKnn, newDataTask)

































