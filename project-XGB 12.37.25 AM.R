names(getModelInfo())

library(Metrics)
library(caret)
library(pROC)
train.data <- read.csv("/Users/zhenqiang/Downloads/train.csv", sep = ",")


for(n in 1:132){
  train.data[,n] <- as.numeric(train.data[,n])
}
train.data$id = NULL


numTrain <- 600
set.seed(4869)
rows_train <- sample(1:nrow(train.data), numTrain)
train <- train.data[rows_train,]


set.seed(4869)
t22 <- train.data[sample(nrow(train.data)),]
split <- floor(nrow(t22)/3)
endata <- t22[0:split,]
bldata <- t22[(split+1):(split*2),]
tedata <- t22[(split*2+1):nrow(t22),]


labelname <- 'loss'
predictors <- names(endata)[names(endata) != labelname]

mycontrol <- trainControl(method = 'cv', number = 10)

model_gbm <- train(endata[,predictors], endata[,labelname], method = 'gbm', trControl = mycontrol)
model_rpart <- train(endata[,predictors], endata[,labelname], method = 'rpart', trControl = mycontrol)
model_xgb <- train(endata[,predictors], endata[,labelname], method = 'xgbTree', trControl = mycontrol)

bldata$gbm_PROB <- predict(object = model_gbm, bldata[,predictors])
bldata$rf_PROB <- predict(object = model_rpart, bldata[,predictors])
bldata$xgb_PROB <- predict(object = model_xgb, bldata[,predictors])

tedata$gbm_PROB <- predict(object = model_gbm, tedata[,predictors])
tedata$rf_PROB <- predict(object = model_rpart, tedata[,predictors])
tedata$xgb_PROB <- predict(object = model_xgb, tedata[,predictors])

#auc <- roc(testingData[,labelName], testingData$gbm_PROB )
#print(auc$auc) # Area under the curve: 0.9893
mae(tedata[,labelname], tedata$gbm_PROB)

#auc <- roc(testingData[,labelName], testingData$rf_PROB )
#print(auc$auc) # Area under the curve: 0.958
mae(tedata[,labelname], tedata$rf_PROB)

#auc <- roc(testingData[,labelName], testingData$treebag_PROB )
#print(auc$auc) # Area under the curve: 0.9734
mae(tedata[,labelname], tedata$xgb_PROB)


predictors <- names(bldata)[names(bldata) != labelname]
final_bl_model <- train(bldata[,predictors], bldata[,labelname], method = 'xgbTree', trControl = mycontrol)

preds_s <- predict(object = final_bl_model, tedata[,predictors])
MAE <- mae(tedata[,labelname], preds_s)
MAE
mae(tedata[,labelname], tedata$xgb_PROB)
mae(tedata[,labelname], tedata$rf_PROB)
mae(tedata[,labelname], tedata$gbm_PROB)

preds_s

submission = data.frame("/Users/zhenqiang/Downloads/sample_submission.csv", colClasses = c("integer", "numeric"))
submission$loss = exp(predict(gbdt,dtest)) - SHIFT
write.csv(submission,'xgb_starter_v7.sub.csv',row.names = FALSE)



submission <- read.csv("/Users/zhenqiang/Downloads/sample_submission.csv", header = T)

submission <- submission[1:5000,]

submission$loss <- preds_s



install.packages("mlbench")
library("caret")
library(mlbench)
library("pROC")
data(Sonar)

set.seed(107)
inTrain <- createDataPartition(y = Sonar$Class, p = .75, list = FALSE)

training <- Sonar[ inTrain,]
testing <- Sonar[-inTrain,]

my_control <- trainControl(
  method="boot",
  number=25,
  savePredictions="final",
  classProbs=TRUE,
  index=createResample(training$Class, 25),
  summaryFunction=twoClassSummary
)

text.data <- read.csv()


test.data <- read.csv ("/Users/zhenqiang/Downloads/test.csv", header = T)
for (n in 1:131) {
  test.data[,n] <- as.numeric(test.data[,n])
}
train.data$id = NULL

labelname <- 'loss'
predictors <- names(test.data)[names(test.data) != labelname]

preds_s <- predict(object = final_bl_model, test.data[,predictors])
submission <- read.csv("/Users/zhenqiang/Downloads/sample_submission.csv", header = T)
submission$loss <- preds_s
write.csv(submission,'FinalResult.csv',row.names = FALSE)





