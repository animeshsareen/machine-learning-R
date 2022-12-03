library(car)
library(tidyverse)
library(ggfortify)
library(vctrs)
library(lmtest)
library(tidyverse)


credit <- read.csv('credit.csv')
lmBack <- step(lm(Balance~., data=credit),
                   direction='backward')

#full model
lmFull <- lm(Balance ~ ., data=credit)
sqrt(
  mean(
    residuals(lmFull)^2
  )
)

#full model rmse 
sqrt(
  mean(
    residuals(lmBack)^2
  )
)

#lmBack rmse
sqrt(
  mean(
    residuals(lmBack)^2
  )
)

#splitting into test/train sets

set.seed(set.seed(82, sample.kind = "Rejection"))
frac <- 0.7
n <- nrow(new_credit)
train.cases <- sample(1:n, frac*n)

train.set <- new_credit[train.cases,]
test.set <- new_credit[-train.cases,]

#re-making and measuring models
model3 <- lm(Balance ~ Income + Limit + Cards + Age + Student, data=train.set)

sqrt(
  mean(
    residuals(model3)^2
  )
)
# full model from this problem
model4 <- lm(Balance ~ ., data=train.set)
sqrt(
  mean(
    residuals(model4)^2
  )
)

sqrt(
  mean(
    (predict(model3, newdata=test.set) - test.set$Balance)^2
  )
)
# full model from this problem

sqrt(
  mean(
    (predict(model4, newdata=test.set) - test.set$Balance)^2
  )
)


#switch to stock market dataset

Smarket <- read.csv('Smarket.csv')
Smarket <- Smarket %>% mutate(Up = ifelse(Direction=='Up', 1, 0))
train.returns <- Smarket %>% filter(Year < 2005)
test.returns <- Smarket %>% filter(Year >= 2005)

logi_full <- glm(Up ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume, data=train.returns, family=binomial)
summary(logi_full)

pred.train <- predict(logi_full, type='response') > 0.5
actual.train <- train.returns$Direction=='Up'
xtabs(~pred.train + actual.train)
mean(pred.train==actual.train)

pred.test <- predict(logi_full, test.returns, type='response') > 0.5
actual.test <- test.returns$Direction=='Up'
xtabs(~pred.test + actual.test)
mean(pred.test==actual.test)

logi_backward <- step(logi_full, direction='backward')
