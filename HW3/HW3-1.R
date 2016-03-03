train.x <- read.table('uspsdata.txt', header = FALSE)
train.y <- read.table('uspscl.txt', header = FALSE)


# The train function
train <- function(X, w, y){
    y <- as.vector(y$V1)
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
    return(c(location_j, theta[location_j], m_list[location_j]))
}