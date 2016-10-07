input <- mtcars[,c('mpg','cyl')]
input <-read.csv("/Users/chris/Downloads/all_10_7.csv",header = TRUE,sep = ',')
print(input)
png(file = "boxplot.png")
data_set <- subset(input,input$project2=="xerces")
boxplot(auc ~ number,data = input,
        xlab="number",
        ylab="auc",
        main="test_20%_xerces")
dev.off()
