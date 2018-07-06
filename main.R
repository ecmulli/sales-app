library(data.table)
library(ggplot2)
library(zoo)
source('Import_and_Explore.R')
train <- import()

test <- train$test
train <- train$train

storespecific <- c('Assortment', 'StoreType', 'AvgSales')
cols <- setdiff(colnames(train), storespecific)
train <- train[,..cols]
storespecific <- c('Assortment', 'StoreType', 'Sales', 'Customers', 'AvgSales')
cols <- setdiff(colnames(test), storespecific)
test <- test[,..cols]

stores <- unique(train$Store)

for ( s in stores){
      sub <- train[Store == s]
      modelcols <- c('Customers','DayOfWeek', 'month', 'Open', 'Promo')
      sub <- sub[,..modelcols]
      modelcols <- colnames(sub)[sub[,lapply(.SD, uniqueN), .SDcols = modelcols] > 1]
      sub <- sub[,..modelcols]
      sub[,Customers := Customers + 1]
      model <- glm(Customers ~ ., data = sub, family = Gamma(link = 'log'))
      train[Store == s, CustPreds := model$fitted.values]
      preds <- exp(predict(model, newdata = test[Store == s]))
      test[Store == s, Customers := preds ]
}

train[, custerr := Customers - CustPreds]
mean(abs(train$custerr))
sqrt(mean(train$custerr^2))

for ( s in stores){
      sub <- train[Store == s]
      modelcols <- c('Sales', 'Customers','DayOfWeek', 'month', 'Open', 'Promo')
      sub <- sub[,..modelcols]
      modelcols <- colnames(sub)[sub[,lapply(.SD, uniqueN), .SDcols = modelcols] > 1]
      sub <- sub[,..modelcols]
      sub[,Sales := Sales + 1]
      model <- glm(Sales ~ ., data = sub, family = Gamma(link = 'log'))
      train[Store == s, SalesPreds := model$fitted.values]
      preds <- exp(predict(model, newdata = test[Store == s]))
      test[Store == s, Sales := preds ]
}

train[, saleserr := Sales - SalesPreds]
mean(abs(train$saleserr))
sqrt(mean(train$saleserr^2))

sub <- train[Store == 262]



ggplot(sub, aes(Sales, fill = DayOfWeek)) + geom_histogram()
ggplot(sub, aes(x = Sales, y = Customers)) + geom_point()
sub <- within(sub, DayOfWeek <- relevel(DayOfWeek , ref = 7))


train <- within(train, DayOfWeek <- relevel(DayOfWeek, ref = 7))
model <- lm(Sales ~ DayOfWeek:month + Open + Promo + Assortment + StoreType, sub)
sumry <- summary(model)
sumry

library(caret)

trainControl <- trainControl(method = "cv",
                             number = 10)
tuneGrid <- expand.grid(
      .alpha=1,
      .lambda=seq(0, 10, by = 0.1))

modelFit <- train(Sales ~ DayOfWeek:month + Open + Promo + Assortment + StoreType, data = train, 
                  method = "glmnet", 
                  trControl = trainControl, # Optimize by F-measure
                  family="gaussian", tuneGrid = tuneGrid)

resp <- train$Sales
reg <- (train[,.(DayOfWeek, month, Open, Promo, Assortment, StoreType)])

cvfit <- cv.glmnet(reg, resp)

model <- train(Sales ~ DayOfWeek:month + Open + Promo + Assortment + StoreType, data=train, method="lm",  trControl=control)
importance <- varImp(model, scale=FALSE)
print(importance)
plot(importance)

library(leaps)
model <- leaps(x = train[,.(Customers)] )
model <- regsubsets(Sales~DayOfWeek + month + Open + Promo + Assortment + StoreType, data = train, method = 'exhaustive', nvmax = 40)
sumry <- summary(model)
sumry$adjr2
which.max(sumry$adjr2)

control <- rfeControl(functions=rfFuncs, method="cv", number=10)

results <- rfe(train[,.(DayOfWeek, month, Open, Promo, Assortment, StoreType)], train[,.(Sales)], sizes=c(1:6), rfeControl=control, nvmax = 20)


library(randomForest)

modelrf <- randomForest(Sales ~ DayOfWeek + month + Open + Promo + Assortment + StoreType,
                        data = train, subset = sample(1:nrow(train), 100000))
preds <- predict(modelrf, train)
err <- data.table(
      sales = train$Sales,
      preds = preds,
      err = train$Sales - preds)
mae <- mean(abs(err))

merr <- melt(err)
ggplot(merr[variable %in% c('sales', 'preds')], aes(variable, value))

ggplot(err[sample(1:1017209, 10000)], aes(sales, preds)) + geom_point()

sparse_matrix <- sparse.model.matrix(Sales ~ .-1, data = train)
