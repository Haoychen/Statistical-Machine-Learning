library(e1071)
# read data
uspsdata <- read.table('uspsdata.txt')
labels <- read.table('uspscl.txt')
labels <- as.factor(labels$V1)

# Split the data set into train data and test data
train_num <- sample(1:dim(uspsdata)[1], 0.8 * dim(uspsdata)[1], replace = F)
test_num <- setdiff(1:dim(uspsdata)[1], train_num)
train.x <- uspsdata[train_num,]
train.y <- labels[train_num]
test.x <- uspsdata[test_num,]
test.y <- labels[test_num]

# train SVM classifier using linear Kernel using 10-fold cross-validation
linear_SVM <- tune(svm, train.x, train.y, kernel = 'linear', ranges = list(cost = c(0.0001, 0.0002, 0.0003, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 1)), tunecontrol = tune.control(sampling = 'cross', cross = 10))

# linear SVM error rate
linear_error <- linear_SVM$performances$error
# cost v.s. error
cost_vs_error <- cbind(c(0.0001, 0.0002, 0.0003, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 1), linear_error)
names(cost_vs_error) <- c('cost', 'error')
# Draw the graph cost v.s. error
plot(cost_vs_error, xlim = c(0, 0.01))

# get the parameter which has the best performance and use it to train a SVM classifier
best_linear_cost <- as.numeric(linear_SVM$best.parameters)
best_linear_SVM <- svm(train.x, train.y, kernel = 'linear', cost = best_linear_cost)

# compute the misclassification error rate based on the test set
linear_predict_y <- predict(best_linear_SVM, test.x)
linear_error_rate <- sum(linear_predict_y != test.y) / length(test.y)

# train SVM classifier using RBF Kernel using 10-fold cross-validation
linear_SVM <- tune(svm, train.x, train.y, kernel = 'radial', ranges = list(cost = 2 ^ (-5: 15), gamma = 2 ^ (-15: 3)), tunecontrol = tune.control(sampling = 'cross', cross = 10))

# linear SVM error rate
linear_error <- linear_SVM$performances$error
# cost v.s. error
cost_vs_error <- cbind(2 ^ (-5: 15), linear_error)
names(cost_vs_error) <- c('cost', 'error')
# Draw the graph cost v.s. error
plot(cost_vs_error, xlim = c(0, 2))

# get the parameter which has the best performance and use it to train a SVM classifier
best_linear_cost <- as.numeric(linear_SVM$best.parameters)
best_linear_SVM <- svm(train.x, train.y, kernel = 'linear', cost = best_linear_cost)

# compute the misclassification error rate based on the test set
linear_predict_y <- predict(best_linear_SVM, test.x)
linear_error_rate <- sum(linear_predict_y != test.y) / length(test.y)

