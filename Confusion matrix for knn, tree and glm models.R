#load packages
library(tibble)
library(dplyr)
library(rpart)
library(rpart.plot)
library(caret)
library(cvms)

#load and prep data

wrk_data = na.omit(as_tibble(read.csv("/Users/safwan/Downloads/NEC/Machine Learning/Case 1 Santander Bank Data/Santander CSVs/train.csv")))

#converting the target column to factor data type
wrk_data$target = as.factor(wrk_data$target)

sapply(wrk_data, class)

wrk_data

unsatisfied_percent = count(wrk_data, "target")


#test-train split
train_idx  = sample(nrow(wrk_data), size = 0.8 * nrow(wrk_data))
train_data = wrk_data[train_idx, ]
test_data  = wrk_data[-train_idx, ]

#200,000 observation have been divided in to 80% training data and 20% test data

#estimation-validation split

estimation_idx = sample(nrow(train_data), size = 0.8 * nrow(train_data))
estimation_data = train_data[estimation_idx, ]
validation_data = train_data[-estimation_idx, ]

#the training data is divided into 80% and 20% or estimation and validation data respectively

head(train_data)
levels(train_data$target)

#as confirmed the target variable has just two levels: "0" - satistfied and "1" - unsatisfied customers

#fitting knn, tree and glm models:

mod_knn = knn3(target ~ var_1 + var_2, data = estimation_data)
mod_tree = rpart(target ~ var_1 + var_2, data = estimation_data)
mod_glm = glm(target ~ var_1 + var_2, data = estimation_data, family = binomial)

#getting predicted probabilites for "1" - unsatisfied customers
set.seed(42)
prob_knn = predict(mod_knn, validation_data) [, "1"]
prob_tree = predict(mod_tree, validation_data) [, "1"]
prob_glm = predict(mod_glm, validation_data, type = "response")

#creating tibble of results for all results
set.seed(42)
results = tibble(
  actual = validation_data$target,
  prob_knn = predict(mod_knn, validation_data) [, "1"],
  prob_tree = predict(mod_tree, validation_data) [, "1"],
  prob_glm = predict(mod_glm, validation_data, type = "response")
)
results

#using cvms package to eveluate various matrices for these models

#knn model
knn_eval_knn = evaluate(
  data = results,
  target_col = "actual",
  prediction_cols = "prob_knn",
  type = "binomial",
  cutoff = 0.5,
  positive = "1",
  metrics = list("Accuracy" = TRUE)
)

#results for eveluation of knn model
knn_eval_knn
knn_eval_knn$Accuracy
knn_eval_knn$Sensitivity
knn_eval_knn$Specificity
knn_eval_knn$Predictions

#confusion matrix plot for knn model
plot_confusion_matrix(knn_eval_knn$`Confusion Matrix` [[1]])

#tree model
knn_eval_tree = evaluate(
  data = results,
  target_col = "actual",
  prediction_cols = "prob_tree",
  type = "binomial",
  cutoff = 0.5,
  positive = "1",
  metrics = list("Accuracy" = TRUE)
)

#results for evaluation of tree model
knn_eval_tree
knn_eval_tree$Accuracy
knn_eval_tree$Sensitivity
knn_eval_tree$Specificity
knn_eval_tree$Predictions

#confusion matrix plot for tree model
plot_confusion_matrix(knn_eval_tree$`Confusion Matrix` [[1]])

#glm model
knn_eval_glm = evaluate(
  data = results,
  target_col = "actual",
  prediction_cols = "prob_glm",
  type = "binomial",
  cutoff = 0.5,
  positive = "1",
  metrics = list("Accuracy" = TRUE)
)

#results for evaluation of glm model
knn_eval_glm
knn_eval_glm$Accuracy
knn_eval_glm$Sensitivity
knn_eval_glm$Specificity
knn_eval_glm$Predictions

#confusion matrix plot for glm model
plot_confusion_matrix(knn_eval_glm$`Confusion Matrix` [[1]])

#Comparing the Confusion matrix from the 3 models tried, only knn model seems to fit. 








