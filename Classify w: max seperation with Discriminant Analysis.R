library(mlr)
library(tidyverse)

install.packages("HDclassif")
data(wine, package = "HDclassif")

wineTib <- as_tibble(wine)
wineTib

wineTib$class <- as.factor(wineTib$class)
wineTib

wineUntidy <- gather(wineTib, "Variable", "Value", -class)

ggplot(wineUntidy, aes(class, Value)) +
facet_wrap(~ Variable, scales = "free_y")  +
geom_boxplot()  +
theme_bw()

wineTask <- makeClassifTask(data = wineTib, target = "class")
lda <- makeLearner("classif.lda")
ldaModel <- train(lda, wineTask)

ldaModelData <- getLearnerModel(ldaModel)
ldaPreds <- predict(ldaModelData)$x
head(ldaPreds)

wineTib %>%
mutate(LD1 = ldaPreds[, 1],
LD2 = ldaPreds[, 2]) %>%
ggplot(aes(LD1, LD2, col = class))  +
geom_point()  +
stat_ellipse() +
theme_bw()

wineUntidy

##Extracting model information using getLearnerMode(), and DF for each case using pred()

ldaModelData <- getLearnerModel(ldaModel)
qda <- makeLearner("classif.qda")
qdaModel <- train(qda, wineTask)

kFold <- makeResampleDesc(method = "RepCV", folds = 10, reps = 50,
stratify = TRUE)

ldaCV <- resample(learner = lda, task = wineTask, resampling = kFold,
measures = list(mmce, acc))

qdaCV <- resample(learner = qda, task = wineTask, resampling = kFold,
measures = list(mmce, acc))

qdaCV <- resample(learner = qda, task = wineTask, resampling = kFold,
measures = list(mmce, acc))

ldaCV$aggr
qdaCV$aggr

qdaCV <- resample(learner = qda, task = wineTask, resampling = kFold,
measures = list(mmce, acc))

qdaCV$aggr

calculateConfusionMatrix(ldaCV$pred, relative = TRUE)

calculateConfusionMatrix(qdaCV$pred, relative = TRUE)

poisoned <- tibble(V1 = 13, V2 = 2, V3 = 2.2, V4 = 19, V5 = 100,
V6 = 2.3, V7 = 2.5, V8 = 0.35, V9 = 1.7,
V10 = 4, V11 = 1.1, V12 = 3, V13 = 750)
predict(qdaModel, newdata = poisoned)
