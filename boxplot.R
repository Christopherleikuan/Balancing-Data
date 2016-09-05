input <- mtcars[,c('mpg','cyl')]
print(head(input))
bug_input <- read.csv("/Users/chris/Downloads/convert.csv",header = ture)
print(head(bug_input[1])
png(file = "boxplot.png")
boxplot(mpg ~ cyl,data = mtcars,
        xlab="Num of cy",
        ylab="miles per",
        main="test")
#dev.off()
