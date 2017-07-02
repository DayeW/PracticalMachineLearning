# PML Project: Prediction Assignment

# Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. 

# Instructions

The project asks to predict the manner in which the subjects did the exercise using the `classe` variable, with focus on variables with `belt, forearm, arm, dumbell`. The report must include how the model is built, how cross validation is used, what the expected out of sample error is, and why I made the choices that I did. I will also use my prediction model to predict 20 different test cases.


# Loading/Processing Data


```r
library(knitr)
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(rpart)
library(rpart.plot)
library(randomForest)
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

First, the testing and training data are loaded into my working directory.


```r
setwd("~/Desktop/Work/Coursera/Prediction")

URLtrain <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(URLtrain, "./train.csv", method = "curl")

URLtest <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(URLtest, "./test.csv", method = "curl")

train <- read.csv("train.csv")
test <- read.csv("test.csv") 
```

Let's look at the data.


```r
head(train[, 1:10], 3) #has NAs <- will need to remove
```

```
##   X user_name raw_timestamp_part_1 raw_timestamp_part_2   cvtd_timestamp
## 1 1  carlitos           1323084231               788290 05/12/2011 11:23
## 2 2  carlitos           1323084231               808298 05/12/2011 11:23
## 3 3  carlitos           1323084231               820366 05/12/2011 11:23
##   new_window num_window roll_belt pitch_belt yaw_belt
## 1         no         11      1.41       8.07    -94.4
## 2         no         11      1.41       8.07    -94.4
## 3         no         11      1.42       8.07    -94.4
```

```r
unique(train$classe) #has levels A B C D E
```

```
## [1] A B C D E
## Levels: A B C D E
```

```r
dim(train); dim(test)
```

```
## [1] 19622   160
```

```
## [1]  20 160
```

In the training data, there are missing values that may affect the accuracy of our predictions, so they will be removed. The `classe` variable also has 5 levels, which indicates what the prediction will look like. 

## Processing Data

First, I'll need to get rid of the near zero covariates - this means that variables with a variance close to zero (no variation) will be deducted from the dataset to produce an accurate model. 


```r
nzcov <- nearZeroVar(train, saveMetrics = TRUE)
train <- train[, !nzcov$nzv]
test <- test[, !nzcov$nzv]
```

Here, I remove most of the missing values from the dataset. Variables consisting of 95% and greater missing values are excluded.  


```r
removeNA <- sapply(train, function(x) mean(is.na(x)) > 0.95)
train <- train[, removeNA == FALSE]
test <- test[, removeNA == FALSE]
head(train[, 1:10], 3)
```

```
##   X user_name raw_timestamp_part_1 raw_timestamp_part_2   cvtd_timestamp
## 1 1  carlitos           1323084231               788290 05/12/2011 11:23
## 2 2  carlitos           1323084231               808298 05/12/2011 11:23
## 3 3  carlitos           1323084231               820366 05/12/2011 11:23
##   num_window roll_belt pitch_belt yaw_belt total_accel_belt
## 1         11      1.41       8.07    -94.4                3
## 2         11      1.41       8.07    -94.4                3
## 3         11      1.42       8.07    -94.4                3
```

The columns left do not have many NAs which will affect our model. However, the first 7 columns contain observations that are not needed for our predictions. These variables will be excluded from the datasets as well.


```r
train <- train[, -(1:7)]
test <- test[, -(1:7)]
dim(train); dim(test)
```

```
## [1] 19622    52
```

```
## [1] 20 52
```

## Partitioning the Data

In order to create our model, I partition the data using the `createDataPartition` function from the `caret` package. This splits the dataset into 60% training and 40% for the test data, as recommended by the course. 


```r
set.seed(12345)
inTrain <- createDataPartition(y = train$classe, p = 0.6, list = FALSE)
training <- train[inTrain, ]
testing <- train[-inTrain, ]
dim(training); dim(testing)
```

```
## [1] 11776    52
```

```
## [1] 7846   52
```

The dimensions of the training and testing data reflect the data partition, and now I am ready to build a prediction model.


# Models for Prediction

In this section, I will build two different prediction models using the Decision Tree algorithm and Random Forest algorithm. Due to system run-time issues, I do not use Boosting, Linear Discriminant Analaysis, or Naive Bayes algorithms to build the predictor. Since using Decision Trees may result in overfitting, I expect that using the Random Forest algorithm may produce a more accurate prediction model, considering the random selection of a subset of observations. For **cross validation**, I use the `confusionMatrix` function to get an estimate of the out of sample error.


## With Decision Tree Modeling

I use the `train` function to fit the data into a prediction model. First I check and plot if using the Decision Tree algorithm may be an accurate predictor for our test data.



```r
modelFit <- train(classe ~., data = training, method = "rpart")
modelFit
modelFit$finalModel
rpart.plot(modelFit$finalModel)
```

![](Project_analysis_files/figure-html/unnamed-chunk-8-1.png)<!-- -->

After seeing the variability in the plot, I considered pruning the plot down to fewer variables; however, I wanted to see instead the accuracy and error rate before continuing on with the Decision Tree algorithm. 


```r
prediction <- predict(modelFit, newdata = testing)
cM <- confusionMatrix(prediction, testing$classe)
cM$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.6027275      0.4927239      0.5917997      0.6135789      0.2844762 
## AccuracyPValue  McnemarPValue 
##      0.0000000      0.0000000
```

```r
cM$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1972  454   98  168  146
##          B   53  686   40   85  311
##          C  157  234 1061  649  355
##          D   46  143  155  384    4
##          E    4    1   14    0  626
```

```r
(error <- 1 - 0.603) #error = 1 - accuracy
```

```
## [1] 0.397
```

The accuracy of using the Decision Tree model is about 0.603, or 60.3%,  which is not very high. Additionally, the error rate is 0.397, calculated by 1 minus the accuracy. This indicates that the Decision Tree model is not an accurate predictor for the test data. 


## With Random Forests

I use the `randomForest` function to fit the training data into a prediction model, since my Decision Tree model does not have high accuracy. 


```r
set.seed(12345)
modelFit1 <- randomForest(classe ~., data = training, method = "class")
modelFit1$finalModel
```

```
## NULL
```


```r
plot(modelFit1)
```

![](Project_analysis_files/figure-html/unnamed-chunk-11-1.png)<!-- -->

```r
prediction1 <- predict(modelFit1, newdata = testing, type = "class")
cM1 <- confusionMatrix(prediction1, testing$classe)
cM1$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.9940097      0.9924221      0.9920420      0.9955953      0.2844762 
## AccuracyPValue  McnemarPValue 
##      0.0000000            NaN
```

```r
cM1$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 2230    9    0    0    0
##          B    2 1504    7    0    0
##          C    0    5 1361   16    2
##          D    0    0    0 1268    4
##          E    0    0    0    2 1436
```

```r
(error1 <- 1 - 0.994) #error = 1 - accuracy
```

```
## [1] 0.006
```

Based off the results, I see that the accuracy using a model built with Random Forest algorithm is 0.994, or 99.4%, which is very high. In addition, the error rate, calculated by 1 minus the accuracy, is 0.006, which is low. Using this model indicates that about 99% of our prediction is accurate, high above the 95% confidence threshold. 


## Conclusion and Predictor Function

In conclusion, the Random Forest algorithm builds an accurate predictor for this dataset, and I use the following to generate predictions for the assignment.


```r
predictionsRF <- predict(modelFit1, newdata = test)
```

