# HW3 Haoyang Chen hc2812
train_x <- read.table('uspsdata.txt', header = FALSE)
train_y <- read.table('uspscl.txt', header = FALSE)


# The train function
train <- function(X, w, y){
    if (!is.vector(y)){
        y <- as.vector(y$V1)
    }
    n <- dim(X)[1]   # the number of observations
    d <- dim(X)[2]   # the number of features
    theta <- c()
    error_rate <- c()
    m_list <- c()
    # compute the error rate for a weak learner
    learning_rule <- function(theta_j, x_j, weight, m_stump){
        predict_y <- c()
        # predict the classification based on the split point
        for (i in c(1: n)){
            if (x_j[i] > theta_j){
                predict_y[i] <- m_stump
            } else {
                predict_y[i] <- -m_stump
            }
        }
        error_rate <- 0
        for (i in c(1: n)){
            if (y[i] != predict_y[i]){
                error_rate <- error_rate + weight[i]
            }
        }
        
        error_rate <- error_rate / (sum(weight))
        return(error_rate)
    }
    
    # compute the optimal parameter theta_j and m for each x
    for (j in c(1: d)){
        test_parameter <- c()
        test_error_rate <- c()
        k <- 1
        # compute the arg min of cost function
        for (m in c(-1, 1)){
            optimization <- optimize(learning_rule, interval = c(min(X[,j]), max(X[,j])), x_j = X[, j], weight = w, m_stump = m)
            test_parameter[k] <- optimization$minimum
            test_error_rate[k] <- optimization$objective
            k <- k + 1
        }
        if(test_error_rate[1] < test_error_rate[2]){
            theta[j] <- test_parameter[1]
            error_rate[j] <- test_error_rate[1]
            m_list[j] <- c(-1)
        } else {
            theta[j] <- test_parameter[2]
            error_rate[j] <- test_error_rate[2]
            m_list[j] <- c(1)
        }
    }
    location_j <- which.min(error_rate)
    result_list = list(j = location_j, theta = theta[location_j], mode = m_list[location_j])
    return(result_list)
}

# The classify function
classify <- function(X, pars){
    label <- (2*(X[, pars$j] > pars$theta) - 1) * pars$mode
    return(label)
}

# The adaBoost function
adaBoost <- function(X, y, B){
    if (!is.vector(y)){
        y <- as.vector(y$V1)
    }
    n <- dim(X)[1]
    d <- dim(X)[2]
    weight_list <- rep(1/n, d)
    alpha <- c()
    allPars <- list()
    
    #train routin
    for (i in c(1: B)){
        allPars[[i]] <- train(X, weight_list, y)
        predict_y <- classify(X, allPars[[i]])
        error_rate <- sum((predict_y != y) * weight_list) / (sum(weight_list))
        alpha[i] <- log((1 - error_rate) / error_rate)
        weight_list <- weight_list * exp(alpha[i] * (predict_y != y))
    }
    
    return(list(alpha = alpha, allPars = allPars))
}

# agg_class function
agg_class <- function(X, alpha, allPars) {
    n <- dim(X)[1]
    B <- length(alpha)
    labels <- matrix(0, nrow = n, ncol = B)
    for (i in c(1: B)){
        labels[,i] <- classify(X, allPars[[i]])
    }
    sum_label <- labels %*% alpha
    classifier <- as.vector(sign(sum_label))
    return(classifier)
}

# cross validation function
cross_validation <- function(X, y, B_max, k_fold){
    n <- dim(X)[1]
    train_error_rate <- matrix(0, nrow = B_max, ncol = k_fold)
    test_error_rate <- matrix(0, nrow = B_max, ncol = k_fold)
    # split data into k group
    num_of_group <- round(n/k_fold)
    k_fold_groups <- list()
    for(k in 1:k_fold){
        ini_point <- (k - 1) * num_of_group
        stop_point <- k * num_of_group
        if(stop_point < n){
            k_fold_groups[[k]] <- c(ini_point: stop_point)
        } else {
            k_fold_groups[[k]] <- c(ini_point : n)
        }
    }
    
    # for each iteration in cross validation, choose train data and test data
    for(k in 1:k_fold){
        train.x <- X[-k_fold_groups[[k]],]
        train.y <- y[-k_fold_groups[[k]],]
        test.x <- X[k_fold_groups[[k]],]
        test.y <- y[k_fold_groups[[k]],]
        ada <- adaBoost(train.x, train.y, B_max)
        allPars <- ada$allPars
        alpha <- ada$alpha
        
        # compute test error rate and train error rate for each b
        for (b in 1: B_max){
            test_predict_y <- agg_class(test.x, alpha[1:b], allPars = allPars[1:b])
            test_error_rate[b, k] <- mean(test.y != test_predict_y)
            train_predict_y <- agg_class(train.x, alpha[1:b], allPars = allPars[1:b])
            train_error_rate[b, k] <- mean(train.y != train_predict_y)
        }
    }
    return(list(train_error_rate = train_error_rate, test_error_rate = test_error_rate))
}

result <- cross_validation(train_x, train_y, 60, 5)
matplot(result$train_error_rate, type = 'l', lty = 1: 5, main = 'Training Error', xlab = 'number of base classifier',
        ylab = 'error rate', ylim = c(0, 0.2))

matplot(result$test_error_rate, type = 'l', lty = 1: 5, main = 'Test Error', xlab = 'number of base classifier',
        ylab = 'error rate', ylim = c(0, 0.2))