# Haoyang Chen
# UNI: hc2812

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

# Use grid method to set tunning parameter cost and gamma
# The range is according to wiki recommendation
costs <- c(0.0001, 0.0005, 0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1)
gammas <- 10 ^ (-4: -1)

# train SVM classifier using linear Kernel using 10-fold cross-validation and record the error v.s each cost
linear_SVM <- c() # record the cost and error rate
for (cost in costs){
    classifier <- svm(train.x, train.y, kernel = 'linear', cost = cost, cross = 10)
    error_rate <- (100 - classifier$tot.accuracy) / 100
    linear_SVM <- rbind(linear_SVM, c(cost, error_rate))
}

# draw the plot
plot(linear_SVM, type = 'b', lty = 2, xlab = 'cost', ylab = 'error rate', main = "Linear SVM error rate")

# select the tunning parameter which has the best performance
# best cost is 0.001
linear_SVM <- as.data.frame(linear_SVM)
names(linear_SVM) <- c('cost', 'error_rate')
best_linear_cost <- as.vector(linear_SVM[linear_SVM$error_rate == min(linear_SVM$error_rate),]$cost)
if(length(best_linear_cost) > 1){
    best_linear_cost <- best_linear_cost[1]
}
# Use the cost with best performance to train a tuning parameter
linear_SVM_model <- svm(train.x, train.y, kernel = 'linear', cost = best_linear_cost)
# Compute the error rate of the linear SVM model using test set
linear_predict_y <- predict(linear_SVM_model, test.x)
linear_error_rate <- sum(linear_predict_y != test.y) / length(test.y)

# train SVM classifier using RBF Kernel using 10-fold cross-validation and record the error v.s each cost and gamma
radial_SVM <- c() # record error v.s. cost and gamma
for (cost in costs){
    for (gamma in gammas){
        classifier <- svm(train.x, train.y, kernel = 'radial', cost = cost, gamma = gamma, cross = 10)
        error_rate <- (100 - classifier$tot.accuracy) / 100
        radial_SVM <- rbind(radial_SVM, c(cost, gamma, error_rate))
    }
}

# draw the plot
radial_SVM <- as.data.frame(radial_SVM)
names(radial_SVM) <- c('cost', 'gamma', 'error_rate')
cols <- c('red', 'blue', 'black', 'green')
i <- 1
for (gamma in gammas){
    col <- cols[i]
    if(i == 1){
        plot(radial_SVM[radial_SVM$gamma == gamma,]$cost, radial_SVM[radial_SVM$gamma == gamma,]$error_rate, type = 'b', lty = 2, col = col, xlim = c(0, 1), ylim = c(0, 0.6), xlab = 'cost', ylab = 'error rate', main = 'RBF SVM error rate')
    }else{
        lines(radial_SVM[radial_SVM$gamma == gamma,]$cost, radial_SVM[radial_SVM$gamma == gamma,]$error_rate, type = 'b', lty = 2, col = col)
    }
    i <- i + 1
}
legend(locator(1), title="gamma", c("10^(-4)","10^(-3)", "10^(-2)", "10^(-1)"),
       lty=2, pch=1, col=cols, cex = 0.5)

# select the tunning parameter which has the best performance
# best cost is 0.5, best gamma is 0.001
best_radial_parameter <- radial_SVM[radial_SVM$error_rate == min(radial_SVM$error_rate),]
if(dim(best_radial_parameter)[1] > 1){
    best_radial_cost <- best_radial_parameter$cost[1]
    best_radial_gamma <- best_radial_parameter$gamma[1]
}
# Use the cost with best performance to train a tuning parameter
radial_SVM_model <- svm(train.x, train.y, kernel = 'radial', cost = best_radial_cost, gamma = best_radial_gamma)
# Compute the error rate of the RBF SVM model using test set
radial_predict_y <- predict(radial_SVM_model, test.x)
radial_error_rate <- sum(radial_predict_y != test.y) / length(test.y)