### Stacked Generalization ###
library(data.table)
library(foreach)
library(MatrixModels)

library(e1071)
library(FNN)
library(glmnet)
library(ranger)
library(xgboost)

# error measure
mse = function(y_dash, y) {
  mse = mean((y - y_dash)^2)
  return(mse)
}

# load and combine dataset
train = fread("BlogFeedback-Train.csv")
test = fread("BlogFeedback-Test.csv")

# create design matrices
test_x = model.Matrix(V281 ~ . - 1, data = test, sparse = F)
test_x_sparse = model.Matrix(V281 ~ . - 1, data = test, sparse = T)
test_xgb = xgb.DMatrix(data = as.matrix(test_x), label = test_y)

train_y = train$V281
test_y = test$V281

# divide training set into k folds
k = 3
cv_index = 1:nrow(train)
cv_index_split = split(cv_index, cut(seq_along(cv_index), k, labels = FALSE))
                 
# meta features from kNN
meta_knn_test = rep(0, nrow(test))
meta_knn_train = foreach(i = 1:k, .combine = c) %do% {

    # split the raining set into two disjoint sets
  train_index = setdiff(1:nrow(train), cv_index_split[[i]])
  train_set1 = model.Matrix(V281 ~ . - 1, data = train[train_index], sparse = T)
  train_set2 = model.Matrix(V281 ~ . - 1, data = train[cv_index_split[[i]]], sparse = T)
  
  # level 0 prediction
  meta_pred = knn.reg(train_set1, train_set2, train[train_index]$V281, k = 19)$pred
  meta_knn_test = meta_knn_test + knn.reg(train_set1, test_x_sparse, train[train_index]$V281, k = 19)$pred / k
  
  return(meta_pred)
}

# meta features from LASSO
meta_glm_test = rep(0, nrow(test))
meta_glm_train = foreach(i = 1:k, .combine = c) %do% {
  
  # split the raining set into two disjoint sets
  train_index = setdiff(1:nrow(train), cv_index_split[[i]])
  train_set1 = model.Matrix(V281 ~ . - 1, data = train[train_index], sparse = T)
  train_set2 = model.Matrix(V281 ~ . - 1, data = train[cv_index_split[[i]]], sparse = T)
  
  # level 0 prediction
  temp_glm = cv.glmnet(train_set1, train[train_index]$V281, family = "gaussian", alpha = 1)
  meta_pred = predict(temp_glm, newx = train_set2)
  meta_glm_test = meta_glm_test + predict(temp_glm, newx = test_x_sparse) / k
  
  return(meta_pred)
}

# meta features from SVM
meta_svm_test = rep(0, nrow(test))
meta_svm_train = foreach(i = 1:k, .combine = c) %do% {
  
  # split the raining set into two disjoint sets
  train_index = setdiff(1:nrow(train), cv_index_split[[i]])
  train_set1 = train[train_index]
  train_set2 = train[cv_index_split[[i]]]
  
  # level 0 prediction
  temp_svm = svm(V281 ~ V52 + V55 + V61 + V51 + V54 + V21 + V6 + V10, data = train_set1, 
                 kernel = "radial", cost = 2, gamma = 0.25)
  meta_pred = predict(temp_svm, train_set2)
  meta_svm_test = meta_svm_test + predict(temp_svm, test) / k
  
  return(meta_pred)
}

# meta features from random forest
meta_rf_test = rep(0, nrow(test))
meta_rf_train = foreach(i = 1:k, .combine = c) %do% {
  
  # split the raining set into two disjoint sets
  train_index = setdiff(1:nrow(train), cv_index_split[[i]])
  train_set1 = train[train_index]
  train_set2 = train[cv_index_split[[i]]]
  
  # level 0 prediction
  temp_rf = ranger(V281 ~ ., data = train_set1, num.trees = 50, mtry = 120, write.forest = T)
  meta_pred = predict(temp_rf, train_set2)$predictions
  meta_rf_test = meta_rf_test + predict(temp_rf, test)$predictions / k
  
  return(meta_pred)
}

# meta features from XGBoost
meta_xgb_test = rep(0, nrow(test))
meta_xgb_train = foreach(i = 1:k, .combine = c) %do% {
  
  # split the raining set into two disjoint sets
  train_index = setdiff(1:nrow(train), cv_index_split[[i]])
  train_set1 = model.Matrix(V281 ~ . - 1, data = train[train_index], sparse = F)
  train_set2 = model.Matrix(V281 ~ . - 1, data = train[cv_index_split[[i]]], sparse = F)
  
  # xgb data
  train_set1_xgb = xgb.DMatrix(data = as.matrix(train_set1), label = train[train_index]$V281)
  train_set2_xgb = xgb.DMatrix(data = as.matrix(train_set2), label = train[cv_index_split[[i]]]$V281)
  
  # level 0 prediction
  temp_xgb = xgboost(data = train_set1_xgb, nround = 750, nthread = 4, max_depth = 6, eta = 0.025, subsample = 0.7, gamma = 3)
  meta_pred = predict(temp_xgb, train_set2_xgb)
  meta_xgb_test = meta_xgb_test + predict(temp_xgb, test_xgb) / k
  
  return(meta_pred)
}
save(meta_knn, file = "meta_knn.stack.rda")  
save(meta_glm, file = "meta_glm.stack.rda")  
save(meta_svm, file = "meta_svm.stack.rda")  
save(meta_rf, file = "meta_rf.stack.rda")  
save(meta_xgb, file = "meta_xgb.astack.rda")  

# combine meta features
sg_col = c("meta_knn", "meta_glm", "meta_svm", "meta_rf", "meta_xgb", "y")
train_sg = data.frame(meta_knn_train, meta_glm_train, meta_svm_train, meta_rf_train, meta_xgb_train, train_y)
test_sg = data.frame(meta_knn_test, meta_glm_test, meta_svm_test, meta_rf_test, meta_xgb_test, test_y)
colnames(train_sg) = sg_col
colnames(test_sg) = sg_col

# ensemble with elastic-net regression
train_sg_sparse = model.Matrix(y ~ . - 1, data = train_sg, sparse = T)
test_sg_sparse = model.Matrix(y ~ . - 1, data = test_sg, sparse = T)

mdl_glm = cv.glmnet(train_sg_sparse, train_y, family = "gaussian", alpha = 0.2)
save(mdl_glm, file = "mdl_glm.ensemble.rda")  

pred_glm = predict(mdl_glm, newx = test_sg_sparse, s = "lambda.min")
mse(pred_glm, test_y)
