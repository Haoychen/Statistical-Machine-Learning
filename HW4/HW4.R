H <- matrix(readBin('histograms.bin', 'double', 640000), 40000, 16)
H <- H + 0.01

# Multinomial EM algorithm
MultinomialEM <- function(H, K, tau) {
    n = dim(H)[1]
    d = dim(H)[2]
    iteration = 1
    
    # Initialize the weights
    c = rep(1, K) / K
    
    # Initialize the centroids
    indices = sample(1: n, K)
    t = H[indices,]
    
    # Iteration
    change = 1
    while (change > tau){
        
        # E-Step
        log_t = log(t)
        Phi = exp(H %*% t(log_t))
        A = matrix(rep(1/rowSums(Phi * c), K), n, K) * (Phi * c)
        
        # M-Step
        t_old = t
        c = colSums(A)
        c = c / sum(c)
        b = t(A) %*% H
        t = matrix(rep(1/apply(b, 1, sum), d), K, d) * b

        
        # Compute changes
        iteration = iteration + 1
        change = norm(t_old - t, type = 'O')
        
    }
    return(I = apply(A, 1, which.max))
}


# Plot the image
imgplot <- function(M_vector){
    img_matrix <- matrix(M_vector, nrow = 200, byrow = T)
    plot_matrix <- NULL
    for (i in 0:(dim(img_matrix)[1] - 1)){
        plot_matrix <- cbind(plot_matrix, img_matrix[dim(img_matrix)[1] - i,])
    }
    image(x = 1:200, y = 1:200, plot_matrix, axes= FALSE, col = grey((0:256) / 256), xlab = '', ylab = '')
}