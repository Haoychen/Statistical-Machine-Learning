# Haoyang Chen
# UNI: hc2812

## Run fakedata function first!
#Inputs
#w:  w[1:d] is the normal vector of a hyperplane, 
#    w[d+1] = -c is the negative offset parameter. 
#n: sample size

#Outputs
#S: n by (d+1) sample matrix with last col 1
#y: vector of the associated class labels

fakedata <- function(w, n){
    
    if(! require(MASS))
    {
        install.packages("MASS")
    }
    if(! require(mvtnorm))
    {
        install.packages("mvtnorm")
    }
    
    require(MASS)
    require(mvtnorm)
    
    # obtain dimension
    d <- length(w)-1
    
    # compute the offset vector and a Basis consisting of w and its nullspace
    offset <- -w[length(w)] * w[1:d] / sum(w[1:d]^2)
    Basis <- cbind(Null(w[1:d]), w[1:d])	 
    
    # Create samples, correct for offset, and extend
    # rmvnorm(n,mean,sigme) ~ generate n samples from N(0,I) distribution
    S <- rmvnorm(n, mean=rep(0,d),sigma = diag(1,d)) %*%  t(Basis) 
    S <- S + matrix(rep(offset,n),n,d,byrow=T)
    S <- cbind(S,1)
    
    # compute the class assignments
    y <- as.vector(sign(S %*% w))
    
    # add corrective factors to points that lie on the hyperplane.
    S[y==0,1:d] <- S[y==0,1:d] + runif(1,-0.5,0.5)*10^(-4)
    y = as.vector(sign(S %*% w))
    return(list(S=S, y=y))
    
} # end function fakedata


# Input: sample data set S, Perceptron weight vector z = (v, -c)
# Output: predict class label vector y
classify <- function(S, z){
    y <- sign(S %*% z)
    y[y == 0] <- 1
    return(y)
}

# Input: sample data set S, class label vector y
#  Output: normal vector Z, the history of the normal vector throught the training run Z_history
perceptrain <- function(S, y){
    dimension <- dim(S)[2] # The dimension of <x, 1>
    n <- dim(S)[1]  # number of observations
    Z <- runif(dimension, min = -1, max = 1) # Start at a random z
    k <- 1 # Initial the iteration time
    Cost <- 100000 # Set initial Perceptron cost function value with a large number
    Z_history <- Z
    while((Cost > 0) || (k < 1000)){
        Cost <- 0 # Set Cost = 0 at the begining of each iteration
        # Set Gradient of the cost function = 0 at the begining of each iteratin
        Gradient_Cost <- rep(0, dimension) 
        for (i in c(1:n)){
            x_vector <- S[i,]
            predict_y <- classify(x_vector, Z)
            if (predict_y != y[i]){
                Cost <- Cost + abs(Z %*% x_vector)
                Gradient_Cost <- Gradient_Cost + (-y[i]) * x_vector
            }
        }
        Z <- Z - (1 / k) * Gradient_Cost
        Z_history <- rbind(Z_history, Z)
        k <- k + 1
    }
    rownames(Z_history) <- c()
    return(list(Z = Z, Z_history = Z_history))
}

# Produce a sample data set using z = c(-10, 6, -5) as train set
perceptron_weight_vector <- c(-3, 5, -2)
S.train <- fakedata(perceptron_weight_vector, 100)
# Build Classifier using perceptron
classifier <- perceptrain(S.train$S, S.train$y)
# Produce a sample data set using z = c(-10, 6, -5) as test set
S.test <- fakedata(perceptron_weight_vector, 100)
# Predict test data response
predict_y <- classify(S.test$S, classifier$Z)
# Calculate error rate
error <- sum(predict_y != S.test$y) / length(S.test$y)
# Convert data to 2D corresponding representation
test.pos <- S.test$y == 1
test.data_pos <- S.test$S[test.pos, 1:2]
test.data_neg <- S.test$S[!test.pos, 1:2]
# Convert the 3D vectors into corresponding line 
vector_to_line <- function (vector){
    Null_space <- Null(vector[1:2])
    offset <- (-vector[3]) * vector[1:2] / (vector[1] ^ 2 + vector[2] ^ 2)
    x1 <- -10 + offset [1]
    x2 <- 10 + offset [1]
    y1 <- -10 * Null_space[2] / Null_space[1] + offset[2]
    y2 <- 10 * Null_space[2] / Null_space[1] + offset[2]
    Hyperplane <- rbind (c(x1, y1), c(x2, y2))
    return (Hyperplane)
}
# Draw dots
plot(test.data_pos, pch = 4, col = 'red', xlim = c(-12, 12), ylim = c(-16, 15), xlab = "", ylab = '')
points(test.data_neg, pch = 1, col = 'blue')

# The line produce the data
origin_line <- vector_to_line(perceptron_weight_vector)
# Classifier Line
classifier_line <- vector_to_line(classifier$Z)
# Draw lines
segments(classifier_line[1, 1], classifier_line[1, 2], classifier_line[2, 1], classifier_line[2, 2], col = 'green')
segments(origin_line[1, 1], origin_line[1, 2], origin_line[2, 1], origin_line[2, 2], col = 'black')

legend(locator(1), title="Boundary", c("Origin Line","Classifier"),
       lty = 1, col=c('black', 'green'), cex = 0.5)

title(main = "Decision Boundary using batch perceptron")


# split the train data into two piece and transfer into 2D points
train.pos <- S.train$y == 1
train.data_pos <- S.train$S[train.pos, 1:2]
train.data_neg <- S.train$S[!train.pos, 1:2]

# Draw dots
plot(train.data_pos, pch = 4, col = 'red', xlim = c(-12, 12), ylim = c(-16, 15), xlab = "", ylab = '')
points(train.data_neg, pch = 1, col = 'blue')

# Transfer the 1st, 100th, 200th ..., 1000th iteration to 2D
history_classifier <- classifier$Z_history[c(1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000),]
for(i in c(1: 10)){
    history_classifier_line <- vector_to_line(history_classifier[i,])
    segments(history_classifier_line[1, 1], history_classifier_line[1, 2], history_classifier_line[2, 1], history_classifier_line[2, 2], col = 'green')
}

# Draw 1000th classifier
history_classifier_line <- vector_to_line(history_classifier[11,])
segments(history_classifier_line[1, 1], history_classifier_line[1, 2], history_classifier_line[2, 1], history_classifier_line[2, 2], col = 'black')

legend(locator(1), title="Iteration", c("Other iteration","1000th Iteration"),
       lty = 1, col=c('green', 'black'), cex = 0.5)

title('Decision Boundaries Over Iterations')