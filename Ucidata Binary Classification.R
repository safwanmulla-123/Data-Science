devtools::install_github("coatless/ucidata")

#loading libraries
library(ucidata)
library(tibble)
library(rpart)
library(rpart)
library(rpart.plot)
library(caret)
library(cvms)

bcw = as_tibble(bcw_original)
bcw

#loading data
bc = na.omit(as_tibble(bcw_original))

bc

#preparing the data

bc = bc %>%
  dplyr::mutate(class = factor(class, labels = c("benign", "malignant"))) %>%
  dplyr::select(-sample_code_number)
bc

#test_train_split
bc_train_idx = sample(nrow(bc), size = 0.8 * nrow(bc))
bc_train = bc[bc_train_idx, ]
bc_test = bc[-bc_train_idx, ]

# estimation-validation split
bc_estimation_idx = sample(nrow(bc_train), size = 0.8 * nrow(bc_train))
bc_estimation = bc_train[bc_estimation_idx, ]
bc_validation = bc_train[-bc_estimation_idx, ]

#checking the data
head(bc_train)
levels(bc_train$class)

#fit models
mod_knn = knn3(class ~ clump_thickness + mitoses, data = bc_estimation)
mod_tree = rpart(class ~ clump_thickness + mitoses, data = bc_estimation)
mod_glm = glm(class ~ clump_thickness + mitoses, data = bc_estimation, family = binomial)

#get predicted probabilities for "positive" class
set.seed(42)
prob_knn = predict(mod_knn, bc_validation) [, "malignant"]
prob_tree = predict(mod_tree, bc_validation) [, "malignant"]
prob_glm = predict(mod_glm, bc_validation, type = "response")

#create tobbles of results
set.seed(42)
results = tibble(
  actual = bc_validation$class,
  prob_knn = predict(mod_knn, bc_validation) [, "malignant"],
  prob_tree = predict(mod_tree, bc_validation) [, "malignant"],
  prob_glm = predict(mod_glm, bc_validation, type = "response")
)
results

#evaluate knn with various mertics
knn_eval = evaluate(
  data = results, target_col = "actual", prediction_cols = "prob_knn",
  type = "binomial", cutoff = 0.5, positive = "malignant",
  metrics = list("Accuracy" = TRUE)
)

#view results of evaluation
knn_eval
knn_eval$Accuracy
knn_eval$Sensitivity
knn_eval$Specificity
knn_eval$Predictions
knn_eval$`Confusion Matrix`
#plot confusion matrix
plot_confusion_matrix(knn_eval$`Confusion Matrix` [[1]])


#evaluate tree with various materics
tree_eval = evaluate(
  data = results, target_col = "actual", prediction_cols = "prob_tree",
  type = "binomial", cutoff = 0.5, positive = "malignant",
  metrics = list("Accuracy" = TRUE)
)

plot_confusion_matrix(tree_eval$`Confusion Matrix` [[1]])


#evaluate glm with various materics
glm_eval = evaluate(
  data = results, target_col = "actual", prediction_cols = "prob_glm",
  type = "binomial", cutoff = 0.5, positive = "malignant",
  metrics = list("Accuracy" = TRUE)
)

plot_confusion_matrix(glm_eval$`Confusion Matrix` [[1]])











