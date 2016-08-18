library (randomForest)
library (ROCR)

set.seed(2016)
#read data
setwd("C:/Users/Chris/Desktop/R/_data") 
#nameList <- read.csv("lang_1.csv", header = FALSE)
parameter <- read.csv("C:/Users/Chris/Desktop/R/rf/parameter_rf.txt", header = TRUE)

#dataName = "ant_promise"
repeatTime = 10

dataName<- "lang"
  auc.final <- 0
  #load the data's parameter
  for (pid in 1:nrow(parameter)) 
  {
    if (parameter[pid,1] == dataName) break;
  }
  train<-read.csv(paste("C:/Users/Chris/Desktop/R/_data/","lang","_0.csv",sep=""),
                  colClasses = c(rep("numeric",29),"factor"),
                  header = TRUE )
  test<-read.csv(paste("C:/Users/Chris/Desktop/R/_data/","lang","_1.csv",sep=""),
                    colClasses = c(rep("numeric",29),"factor"),
                    header = TRUE )
    
  #repeat several times, use the average as the result
  cat(paste("doing",dataName,Sys.time(),'\n',sep = '\t'))#print progress
  for (repeatID in 1:repeatTime)
  {
    #generate rf
    rf <- randomForest(bug ~.,
                       data = train,
                       ntree = min(3000,nrow(train)),
                       mtry = parameter[pid,'mtry'],
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
                file = paste("C:/Users/Chris/Desktop/R/rf/repeat0/","lang","_",repeatID,".csv",sep = ""),
                row.names = FALSE,
                sep = ',')
    auc.final <- auc.final + auc
  }
  #print the score of each module
  wriwomte(paste(0,dataName,auc.final/repeatTime,sep = ","),
        file = "C:/Users/Chris/Desktop/R/rf/result.txt",
        append = TRUE)


