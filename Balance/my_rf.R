library (randomForest)
library (ROCR)
set.seed(2016)
setwd("C:/Users/Chris/Desktop/R/_data") #读文件
#parameter <- read.csv("C:/Users/Chris/Desktop/R/rf/parameter_rf.txt", header = TRUE) #参数文件

traing <- function(test,train,variable){
  auc.final <- 0
  repeatTime = 10
  for (repeatID in 1:repeatTime)
  {
    #generate rf
    rf <- randomForest(bug ~.,
                       data = train,
                       ntree = min(3000,nrow(train)),
                     #  mtry = parameter[pid,'6'],
                       keep.forest = TRUE, 
                       importance = TRUE
    )
    #predict the cv test set
    rf.pr = predict(rf, type="prob",newdata = test)[,2]
    rf.pred = prediction(rf.pr, test$bug)
    rf.perf = performance(rf.pred, "tpr", "fpr")
    #calculate AUC
    auc <- performance(rf.pred, "auc")
    auc <- unlist(slot(auc, "y.values"))
    #write the result
    result<-data.frame(test = c(unname(rf.pr)),
                       condition = c(test$bug))
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
dataName<- list("lang","math","time")
for (variable in dataName) {


  
  filetest <- dir(variable)
  filetestname <- filetest[length(filetest)]
  filetestname <- substr(filetestname,1,nchar(filetestname)-4) # 将文件名提取出来，将.csv后缀去掉，其实多此一举。
  
  test<-read.csv(paste("C:/Users/Chris/Desktop/R/_data/",variable,"/",filetestname,".csv",sep=""),
                  colClasses = c(rep("numeric",29),"factor"),
                  header = TRUE )
  print("11111")
  #至此已经把测试集挑出
  #下一步是找到训练集文件然在找到他的4个算法子文件
  filetrain_name <- filetest[length(filetest)-1]  #相邻的数据集
  filetrain_name <- substr(filetrain_name,1,nchar(filetrain_name)-4) #提取没有后缀的文件名
  train_parent <- read.csv(paste("C:/Users/Chris/Desktop/R/_data/",variable,"/",filetrain_name,".csv",sep=""),
                           colClasses = c(rep("numeric",29),"factor"),
                           header = TRUE)
  traing(test,train_parent,variable)
  print("22222")
  #读取算法子数据集
  file_new <- paste("C:/Users/Chris/Desktop/R/nodeal_data/result_new/result/",variable,sep = "" ) #存放算法子文件的文件夹
  file_new_list <- dir(file_new)
  for (variable_1 in file_new_list) {        #在算法子数据集文件夹中匹配相应的算法子
    train_name <- substr(variable_1,1,nchar(variable_1)-4)
    if (paste(filetrain_name,"-0",sep = "")==train_name) {
      train_1 <- read.csv(paste("C:/Users/Chris/Desktop/R/nodeal_data/result_new/result/",variable,"/",train_name,".csv",sep=""),
                               colClasses = c(rep("numeric",29),"factor"),
                               header = TRUE)
      traing(test,train_1,variable)
      print("3333")
    }
    else if (paste(filetrain_name,"-1",sep = "")==train_name) {
      train_2 <- read.csv(paste("C:/Users/Chris/Desktop/R/nodeal_data/result_new/result/",variable,"/",train_name,".csv",sep=""),
                          colClasses = c(rep("numeric",29),"factor"),
                          header = TRUE)
      traing(test,train_2,variable)
      print("4444")
    }
    else if (paste(filetrain_name,"-2",sep = "")==train_name) {
      train_3 <- read.csv(paste("C:/Users/Chris/Desktop/R/nodeal_data/result_new/result/",variable,"/",train_name,".csv",sep=""),
                          colClasses = c(rep("numeric",29),"factor"),
                          header = TRUE)
      traing(test,train_3,variable)
      print("5555")
    }
    else if (paste(filetrain_name,"-3",sep = "")==train_name) {
      train_4 <- read.csv(paste("C:/Users/Chris/Desktop/R/nodeal_data/result_new/result/",variable,"/",train_name,".csv",sep=""),
                          colClasses = c(rep("numeric",29),"factor"),
                          header = TRUE)
      traing(test,train_4,variable)
      print("666")
    }
  }
  
  
}


