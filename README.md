# Machine Learning in R: Stepwise Selection

## Project ## 
Applying stepwise regression + selection techniques to optimize predictive accuracy.

Here, we are using stepwise selection + ML to optimize our financial predictions.

E.g. `expected credit balance`, `daily market direction`, etc.
___
## Background ## 
- You’re ramping up to start a Machine Learning internship with the Synergy Group this summer. They’re one of the biggest names in asset + lending services, and have a strong financial reputation for maximizing predictive success. 
___
## Problem ##
After reading all about your recent Data Science experience at UT Austin, your manager Michael is eager to use your skills to improve the company's bottom line. 
___
Data - `credit.csv` and `new_credit.csv`

* `Income`: Income in $1,000's
* `Limit`: Credit limit
* `Rating`:Credit rating
* `Cards`: Number of credit cards
* `Age`: Age in years
* `Education`: Education in years
* `Own`: Whether the individual owns a home
* `Student`: Whether the individual is a student
* `Married`: Whether the individual is married
* `Region`: A factor with levels East, South, and West indicating the individual's geographical location
* `Balance`: Average credit card balance in $

Other data - 'Smarket.csv' 
* `Year`: The year that the observation was recorded
* `Lag1`: Percentage return for previous day
* `Lag2`: Percentage return for 2 days previous
* `Lag3`: Percentage return for 3 days previous
* `Lag4`: Percentage return for 4 days previous
* `Lag5`: Percentage return for 5 days previous
* `Volume`: Volume of shares traded (number of daily shares traded in billions)
* `Today`: Percentage return for today
* `Direction`: A categorical variable with levels `Down` and `Up` indicating whether the market had a positive or negative return on a given day

## Deliverable ## 

Apply stepwise selection to create accurate models and predict financial outcomes. But our model should be `reliable` in its predictions of credit card `Balance.`

Reliable simply means consistent, especially between different samples.

## Starting Somewhere

---

```r
# starting with backwards stepwise

credit <- read.csv('credit.csv')
lmBack <- step(lm(Balance~., data=credit),
                   direction='backward')
```

With little idea where to start, I elected to include all predictors
and perform backward stepwise regression to derive the ideal model. 

Here’s a table with my results:

| Step | Removed Variable | Resulting AIC |
| --- | --- | --- |
| 1 | Own | 1400.7 |
| 2 | Education | 1399.0 |
| 3 | Rating | 1397.3 |
| 4 | Married | 1396.1 |
| 5 | Region | 1395.1 |

Next, I compared the accuracy of that model to a full model (with all-predictors) — and the results proved my initial thinking right.

```r
#solving for RMSEs

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
```

### Why Accuracy Isn’t Accurate (in this case)

---

The model with the highest accuracy isn’t our backwards-selected one, but actually the full model (as it has the lower RMSE of the two.)

So why is arbitrarily stuffing everything into our model more accurate than using backwards selection, relative to predicting `Balance`  ?

When you consider how our error metric of RMSE was computed, you realize it isn’t a fair measure of true predictive power at all. The RMSE was calculated on data that the model was trained on. That’s why it may not generalize to new data that our model has never seen before. 

I would split the data into training + test sets, then try again from there.

### Splitting, Training, Testing

---

Now we can use the other dataset  `new_credit.csv` 

Here’s some code I found to split a sample into training/test sets.

```r
set.seed(set.seed(82, sample.kind = "Rejection"))
frac <- 0.7
n <- nrow(new_credit)
train.cases <- sample(1:n, frac*n)

train.set <- new_credit[train.cases,]
test.set <- new_credit[-train.cases,]

```

We then sample the initial model code we used. 

```r
model3 <- lm(Balance ~ Income + Limit + Cards + Age + Student, data=train.set)

sqrt(
  mean(
    residuals(model3)^2
  )
)
# full model
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
# full model

sqrt(
  mean(
    (predict(model4, newdata=test.set) - test.set$Balance)^2
  )
)
```

Here, the more accurate model (when judging based on test set RMSE) is our backward selection one. The full model likely suffered from overfitting to the training data + couldn’t translate its predictive policy to another sample.

### Evaluating Categorical Predictors

---

Now load the `Smarket` data in `Smarket.csv`. This dataset contains daily percentage returns of the S&P 500 for the years 2001 to 2005. The variables in the data set are:

- `Year`: The year that the observation was recorded
- `Lag1`: Percentage return for previous day
- `Lag2`: Percentage return for 2 days previous
- `Lag3`: Percentage return for 3 days previous
- `Lag4`: Percentage return for 4 days previous
- `Lag5`: Percentage return for 5 days previous
- `Volume`: Volume of shares traded (number of daily shares traded in billions)
- `Today`: Percentage return for today
- `Direction`: A categorical variable with levels `Down` and `Up` indicating whether the market had a positive or negative return on a given day

Try and predict market direction using stepwise selection.
    
    Why not split into `training/test set` like before?
    
    Because back then, our observations were intrinsically invariant to time; their relative position and order within the data didn’t matter to predictive interests.
    
    Now, however, we’ve gotten a time series of information, which implies that we’ll necessarily need to evaluating every point in the historical context of its local data.
    
     (i.e. that’s what seasonality and trend components are doing in time series forecasting)
    

```r
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
```
