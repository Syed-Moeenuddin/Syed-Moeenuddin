################################ CLASSIFICATION ####################################
####################################################################################

#import dataset

df <- read.csv("C:/Users/ADMIN/Desktop/RStudio work/DSP 34 ew batch/Movie_classification.csv")
View(df)

#Data Preprocessing
summary(df)
summary(df$Time_taken)
df$Time_taken[is.na(df$Time_taken)] <- mean(df$Time_taken,na.rm = TRUE)
summary(df$Time_taken)
# Test-Train Split
#install.packages('caTools')
library(caTools)
set.seed(0)
split =sample.split(df,SplitRatio = 0.8)
trainc = subset(df,split == TRUE)
testc = subset(df,split == FALSE)

#install required packages
install.packages('rpart')
install.packages('rpart.plot')
library(rpart)
library(rpart.plot)

#Run Classification tree model on train set
classtree <- rpart(formula = Start_Tech_Oscar~., data = trainc, method = 'class', control = rpart.control(maxdepth = 3))
summary(classtree)#press F1 on rpart for help on this function
classtree
#Plot the decision Tree
rpart.plot(classtree, box.palette="RdBu", digits = -3)

#Predict value at any point
testc$pred <- predict(classtree, testc, type = "class")

table(testc$Start_Tech_Oscar,testc$pred)

###############################
###  RANDOM FOREST
###############################
#Random forest
install.packages('randomForest')
library(randomForest)

randomfor <- randomForest(Collection~., data = trainc,ntree=500)

#Predict Output 
#Predict value at any point
testc$pred <- predict(randomfor, testc, type = "class")

table(testc$Start_Tech_Oscar,testc$pred)
