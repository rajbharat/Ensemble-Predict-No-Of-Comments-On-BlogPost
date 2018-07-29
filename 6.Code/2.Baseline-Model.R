### Basic Models ###
library(data.table)
library(MatrixModels)

library(e1071)
library(FNN)
library(glmnet)
library(ranger)
library(xgboost)


# error measure
mse = function(ydash, y) {
  mse = mean((y - ydash)^2)
  return(mse)
}

# load and combine dataset
train = fread("BlogFeedback-Train.csv")
test = fread("BlogFeedback-Test.csv")


# create design matrices
train_x = model.Matrix(V281 ~ . - 1, data = train, sparse = F)
train_x_sparse = model.Matrix(V281 ~ . - 1, data = train, sparse = T)

train_y = train$V281

test_x = model.Matrix(V281 ~ . - 1, data = test, sparse = F)
test_y = test$V281

train_xgb = xgb.DMatrix(data = as.matrix(train_x), label = train_y)
test_xgb = xgb.DMatrix(data = as.matrix(test_x), label = test_y)

# try kNN
mdl_knn= knn.reg(train_x, test_x, train_y, k = 15)
pred_knn = knn.reg(train_x, test_x, train_y, k = 15)$pred
mse(pred_knn, test_y)
save(mdl_knn, file = "mdl_knn.rda")

# try LASSO
mdl_lasso = cv.glmnet(train_x_sparse, train_y, family = "gaussian", alpha = 2)
pred_lasso = predict(mdl_lasso, newx = test_x)
mse(pred_lasso, test_y)
save(mdl_lasso, file = "mdl_lasso.rda")

# try SVM
mdl_svm = svm(V281 ~ V52 + V55 + V61 + V51 + V54 + V21 + V6 + V10, data = train, kernel = "radial", cost = 2, gamma = 0.25)
pred_svm = predict(mdl_svm, test)
mse(pred_svm, test_y)
save(mdl_svm, file = "mdl_svm.rda")


# try random forest
mdl_rf = ranger(V281 ~ ., data = train, num.trees = 1000, mtry = 120, write.forest = T)
pred_rf = predict(mdl_rf, test)
mse(pred_rf$predictions, test_y)
save(mdl_lasso, file = "mdl_rf.rda")


# try XGboost
mdl_xgb = xgboost(data = train_xgb, nround = 500, nthread = 4, max_depth = 6, eta = 0.025, subsample = 0.7, gamma = 3)
pred_xgb = predict(mdl_xgb, test_xgb)
mse(pred_xgb, test_y)
save(mdl_xgb, file = "mdl_xgb.rda")