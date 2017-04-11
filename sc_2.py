library (randomForest)
library (ROCR)
library(e1071)
library(DMwR)
library(readr)
set.seed(2016)
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

setwd("C:/Users/Chris/Desktop/") #读文件
#dataName <- list("lang","math","time")
dataName <- list("nasa")
for (variable in dataName) {
  testlist <- dir(variable)
  for (variable_test in testlist) {
    print(variable_test)
    print("111")
    dataset <- read.csv(paste("C:/Users/Chris/Desktop/",variable,"/",variable_test,sep = ""),
                         header = TRUE)
    print("222")
    #找到了原始数据集,下一步是抽出做无监督用的数据集
    datatrain <- dataset[1:length(dataset) - 1]    
    answer <- spectral_clustering_based_classifier(datatrain)
    print("333")
    for (variable_an in length(answer)) {
      print(answer[variable_an])
      if (answer[variable_an] == FALSE) {
        answer[variable_an] <- 0
      }
      else(answer[variable_an] == TRUE)
      answer[variable_an] <- 1
    }
    print("444")
    g <- prediction(answer,dataset$bugs)
    auc <- performance(g,"auc")@y.values
    print(auc)
    auc_value <- unlist(auc)
    csv_out_list <- data.frame(auc_value)
    #csv_out_list <- data.matrix(auc_value)
    write.table(csv_out_list,
                file = paste("C:/Users/Chris/Desktop/",variable,".csv",sep = ""),
                row.names = FALSE,
                append = TRUE,      
                sep = ','
    )
    print("555")
   
  }
 
  
}
