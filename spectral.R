library(ROCR)
spectral_clustering_based_classifier <- function (A) {
# Normalize software metrics .
normA = apply (A, 2, function (x){(x- mean (x))/sd(x)})
# Construct the weighted adjacency matrix W.
W = normA %*% t( normA )
# Set all negative values to zero .
W[W <0] = 0
# Set the self - similarity to zero .
W = W - diag ( diag (W))
# Construct the symmetric Laplacian matrix Lsym .
Dnsqrt = diag (1/ sqrt ( rowSums (W)))
I = diag (rep (1, nrow (W)))
Lsym = I - Dnsqrt %*% W %*% Dnsqrt
# Perform the eigendecomposition .
ret_egn = eigen (Lsym , symmetric = TRUE )
# Pick up the second smallest eigenvector .
v1 = Dnsqrt %*% ret_egn $ vectors [, nrow (W) -1]
v1 = v1 / sqrt ( sum (v1 ^2) )
# Divide the data set into two clusters .
defect_proneness = (v1 >0)
# Label the defective and clean clusters .
rs = rowSums ( normA )
if( mean (rs[v1 >0]) <mean (rs[v1 <0]) )
defect_proneness = (v1 <0)
# Return the defect proneness .
defect_proneness
}

data_matri <- read.csv("C:/Users/Chris/Desktop/3.1.5_new.csv",header = TRUE)
answer <- spectral_clustering_based_classifier(data_matri)


for (variable in length(answer)) {
  print(answer[variable])
  if (answer[variable] == FALSE) {
    answer[variable] <- 0
  }
  else(answer[variable] == TRUE)
  answer[variable] <- 1
}
bugs <- answer
answer <- cbind(data_matri,bugs)
write.table(answer,
            file = "C:/Users/Chris/Desktop/3.1.5_new_33.csv",
            row.names = FALSE,
            sep = ',')

bugs <-  read.csv("C:/Users/Chris/Desktop/bugs_test.csv",header = TRUE)
g <- prediction(answer$bugs,bugs$bugs)
print(g)
perf <- performance(g,"rec","fpr")
plot(perf)
auc <- performance(g,"recall")@y.values
