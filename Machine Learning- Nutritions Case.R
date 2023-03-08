# Week 9 Assignment

library(mlr)
library(tidyverse)
library(kknn)

nutri <- read.csv("/Users/safwan/Downloads/NEC/Machine Learning/Case 2 Nutrition/Final_Nutrition2csv.csv")

nutriTib <- as_tibble(nutri)

colnames(nutriTib)

nutriClean = na.omit(nutriTib)

nutriMutate <- mutate_all(nutriClean, as.numeric) %>%
  select(c(-"Shrt_Desc", -"Long_Desc", -"FdGrp_Desc",
           -"GmWt_Desc1", -"GmWt_Desc2"))

ggplot(nutriMutate, aes(Sugar_Tot_g, Energ_Kcal))  +
  geom_point()  +
  geom_smooth() +
  theme_bw()

### Since Energy Calories would make the most sense to check the data when compared with Sugar, we plot for Energy vs Sugar. And, as expected we get a propotional relationship. Calories incerase with sugar content.

# Building kNN regression model

nutriTask <- makeRegrTask(data = nutriMutate, target = "Energ_Kcal")

kknn <- makeLearner("regr.kknn")

kknnParamSpace <- makeParamSet(makeDiscreteLearnerParam("k", values = 1:12))

gridSearch <- makeTuneControlGrid()

kFold <- makeResampleDesc("CV", iters = 2)

tunedK <- tuneParams(kknn, task = nutriTask,
                     resampling = kFold,
                     par.set = kknnParamSpace,
                     control = gridSearch)
tunedK

tunedKnn <- setHyperPars(makeLearner("regr.kknn"), par.vals = tunedK$x)

tunedKnnModel <- train(tunedKnn, nutriTask)

# Building Random Forest Regression Model

forest <- makeLearner("regr.randomForest")

forestParamSpace <- makeParamSet(
  makeIntegerParam("ntree", lower = 50, upper = 50),
  makeIntegerParam("mtry", lower = 100, upper = 367),
  makeIntegerParam("nodesize", lower = 1, upper = 10),
  makeIntegerParam("maxnodes", lower = 5, upper = 30))
randSearch <- makeTuneControlRandom(maxit = 100)
library(parallel)
library(parallelMap)
parallelStartSocket(cpus = detectCores())
tunedForestPars <- tuneParams(forest, task = nutriTask,
                              resampling = kFold,
                              par.set = forestParamSpace,
                              control = randSearch)
parallelStop()
tunedForestPars

tunedForest <- setHyperPars(forest, par.vals = tunedForestPars$x)
tunedForestModel <- train(tunedForest, nutriTask)
forestModelData <- getLearnerModel(tunedForestModel)
plot(forestModelData)

## It looks like the out-of-bag error stabilizes after 40-50 bagged trees, so we can be satisfied that we have included enough trees in our forest.

# Building XGBoost Regression model

xgb <- makeLearner("regr.xgboost")

xgbParamSpace <- makeParamSet(
  makeNumericParam("eta", lower = 0, upper = 1),
  makeNumericParam("gamma", lower = 0, upper = 10),
  makeIntegerParam("max_depth", lower = 1, upper = 20),
  makeNumericParam("min_child_weight", lower = 1, upper = 10),
  makeNumericParam("subsample", lower = 0.5, upper = 1),
  makeNumericParam("colsample_bytree", lower = 0.5, upper = 1),
  makeIntegerParam("nrounds", lower = 30, upper = 30))
tunedXgbPars <- tuneParams(xgb, task = nutriTask,
                           resampling = kFold,
                           par.set = xgbParamSpace,
                           control = randSearch)
tunedXgbPars

tunedXgb <- setHyperPars(xgb, par.vals = tunedXgbPars$x)
tunedXgbModel <- train(tunedXgb, nutriTask)
xgbModelData <- getLearnerModel(tunedXgbModel)
ggplot(xgbModelData$evaluation_log, aes(iter, train_rmse)) +
  geom_line() +
  geom_point() +
  theme_bw()

### We can see that 30 iterations/trees is just about enough for the RMSE to have flattened out (including more iterations won’t result in a better model).

kknnWrapper <- makeTuneWrapper(kknn, resampling = kFold,
                               par.set = kknnParamSpace,
                               control = gridSearch)
forestWrapper <- makeTuneWrapper(forest, resampling = kFold,
                                 par.set = forestParamSpace,
                                 control = randSearch)
xgbWrapper <- makeTuneWrapper(xgb, resampling = kFold,
                              par.set = xgbParamSpace,
                              control = randSearch)
learners = list(kknnWrapper, forestWrapper, xgbWrapper)
holdout <- makeResampleDesc("Holdout")
bench <- benchmark(learners, nutriTask, holdout)
bench

## Our extimates show that after tuning the hyperparameters XGBoost performance was the best.

































