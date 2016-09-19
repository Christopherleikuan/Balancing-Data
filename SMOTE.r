library(DMwR)
library(readr)
setwd("C:/Users/Chris/Desktop/R/r100_new/result/SMOTE") #读文件
datalist<- list("ant","camel","ivy","jedit","poi","xalan","xerces")
#datalist<- list("ant")
for (variable in datalist) {paste("C:/Users/Chris/Desktop/R/_data/",variable,"/",filetestname,".csv",sep="")
  filetest <- dir(variable)
  filetestname <- filetest[length(filetest)]
  filetestname <- substr(filetestname,1,nchar(filetestname)-4)
  data_test <- read.csv(paste("C:/Users/Chris/Desktop/R/r100_new/result/SMOTE/",variable,"/",filetestname,".csv",sep = ""),
                        colClasses = c(rep("numeric",20),"factor"),
                        header = TRUE )
  table(data_test$bugs)
  #计算需要生成的bugs数
  data_num = table(data_test$bugs)
  bugs = data_num[2]
  all = data_num[1] + data_num[2]
  rate <- 0.6
  make_bug_num = (rate * all - bugs) / (1 - rate)
  make_perc_over = (((bugs + make_bug_num) / bugs ) - 1) * 100
  make_unbug_num = ((((1 - rate) * bugs) + ((1 - rate) * make_bug_num)) / (rate * make_bug_num)) * 100
  count = 0 
  
  
  while (count < 100 ) {
    #SMOTE中的参数 prec.over 是指参入少数例的百分比，比如原有60个数据，那么prec.over = 100 就是生成包含120bug的文件
    new_data =  SMOTE(bugs~.,data_test,perc.over = make_perc_over ,perc.under = make_unbug_num)
    table(new_data$bugs)
    write.table(new_data,
                file = paste("C:/Users/Chris/Desktop/R/r100_new/result/SMOTE/",variable,"/",filetestname,"-",count,".csv",sep = ""),
                row.names = FALSE,
                sep = ',')
    
    count = count + 1
  }
  #prop.table(table(data_test$bugs))
}









