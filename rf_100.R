library (randomForest)
library (ROCR)
library(e1071)
set.seed(2016)
setwd("C:/Users/Chris/Desktop/R/r100_new") #读文件
traing_rf <- function(test,train,variable){
  auc.final <- 0
  repeatTime = 10
  for (repeatID in 1:repeatTime)
  {
    #generate rf
    rf <- randomForest(bugs ~.,
                       data = train,
                       ntree = min(3000,nrow(train)),
                       #  mtry = parameter[pid,'6'],
                       keep.forest = TRUE, 
                       importance = TRUE
    )
    #predict the cv test set
    rf.pr = predict(rf, type="prob",newdata = test)[,2]
    rf.pred = prediction(rf.pr, test$bugs)
    rf.perf = performance(rf.pred, "tpr", "fpr")
    #calculate AUC
    auc <- performance(rf.pred, "auc")
    auc <- unlist(slot(auc, "y.values"))
    #write the result
    result<-data.frame(test = c(unname(rf.pr)),
                       condition = c(test$bugs))
    write.table(result,
                file = paste("C:/Users/Chris/Desktop/R/rf/repeat0/",variable,"_",repeatID,".csv",sep = ""),
                row.names = FALSE,
                sep = ',')
    auc.final <- auc.final + auc
  }
  #print the score of each module
  result_save <- paste("C:/Users/Chris/Desktop/R/rf/",variable,".csv",sep = "")
  write(paste(0,variable,auc.final/repeatTime,sep = ","),
        file = result_save ,
        append = TRUE)
}
#dataName<- list("jedit","poi","xalan","xerces")
dataName <- list("ivy")
for (variable in dataName) {
  testlist <- dir("test")
  
  #找到文件夹中的指定数据集作为测试集,并且在这个循环中把原始数据集给挑出并且训练
  for (variable_test in testlist) {
    if (substr(variable_test,1,nchar(variable_test)-10) == variable) { 
      test<-read.csv(paste("C:/Users/Chris/Desktop/R/r100_new/test/",variable_test,sep=""),
                     colClasses = c(rep("numeric",20),"factor"),
                     header = TRUE )
    }
  }
  trainglist <- dir("origin")
  for (variable_train in trainglist) {
    if (substr(variable_train,1,nchar(variable_train)-10) == variable) { 
      train <- read.csv(paste("C:/Users/Chris/Desktop/R/r100_new/origin/",variable_train,sep=""),
                     colClasses = c(rep("numeric",20),"factor"),
                     header = TRUE )
    }
  }
  traing_rf(test,train,variable)
  print("11111")
  #循环100次读入训练集,这是自己的4总算法子
  traing_file_list <- list("0","1","2","3")
  for (variable_train in traing_file_list) {
    count_0 <- 0
    train_list_0 <- dir(paste("result/",variable,"/",variable_train,"/",sep=""))
    
    print("1111-1")
    while (count_0 < 100) {
      train_0 <- read.csv(paste("C:/Users/Chris/Desktop/R/r100_new/result/",variable,"/",variable_train,"/",train_list_0[count_0+1],sep=""),
                          colClasses = c(rep("numeric",20),"factor"),
                          header = TRUE)
      print("1111-2")
      count_0 <- count_0 + 1
      traing_rf(test,train_0,variable)
      
      
    }
  }
  print("22222")
  
  #循环100次读入训练集，这是别人的3总算法
  traing_file_3_list <- list("add","delete","morph")
  for (variable_3 in traing_file_3_list) {
    count_1 <- 0
    train_list_1 <- dir(paste("result/",variable_3,"/",variable,sep = ""))
    while (count_1 < 100) {
      train_1 <- read.csv(paste("C:/Users/Chris/Desktop/R/r100_new/result/",variable_3,"/",variable,"/",train_list_1[count_1+1],sep=""),
                          colClasses = c(rep("numeric",20),"factor"),
                          header = TRUE)
      count_1 <- count_1 + 1
      traing_rf(test,train_1,variable)
    
    }
  }
  print("33333")
}

