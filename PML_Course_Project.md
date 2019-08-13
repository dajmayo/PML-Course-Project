---
title: "Practical Machine Learning Course Project"
author: "Danna Ashley J. Mayo"
date: "August 13, 2019"
output: 
  html_document:
    keep_md: yes
---



### Project Overview 
This project aims to predict the manner in which the 6 participants from the dataset did their exercise. Here you will see how the model is built, how cross validation is used for the model, what the possible out of sample error would be, and why the following choices are made. Afterwhich, the prediction model built will be used to predict 20 different test cases. 

#### Data Preparation

```r
library(caret)
library(randomForest)
library(dplyr)
library(e1071)
library(ggplot2)

train <- read.csv("pml-training.csv", na.strings = c("NA", "#DIV/0!", ""))
test <- read.csv("pml-testing.csv", na.strings = c("NA", "#DIV/0!", ""))
str(head(train))
```

```
## 'data.frame':	6 obs. of  160 variables:
##  $ X                       : int  1 2 3 4 5 6
##  $ user_name               : Factor w/ 6 levels "adelmo","carlitos",..: 2 2 2 2 2 2
##  $ raw_timestamp_part_1    : int  1323084231 1323084231 1323084231 1323084232 1323084232 1323084232
##  $ raw_timestamp_part_2    : int  788290 808298 820366 120339 196328 304277
##  $ cvtd_timestamp          : Factor w/ 20 levels "02/12/2011 13:32",..: 9 9 9 9 9 9
##  $ new_window              : Factor w/ 2 levels "no","yes": 1 1 1 1 1 1
##  $ num_window              : int  11 11 11 12 12 12
##  $ roll_belt               : num  1.41 1.41 1.42 1.48 1.48 1.45
##  $ pitch_belt              : num  8.07 8.07 8.07 8.05 8.07 8.06
##  $ yaw_belt                : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4
##  $ total_accel_belt        : int  3 3 3 3 3 3
##  $ kurtosis_roll_belt      : num  NA NA NA NA NA NA
##  $ kurtosis_picth_belt     : num  NA NA NA NA NA NA
##  $ kurtosis_yaw_belt       : logi  NA NA NA NA NA NA
##  $ skewness_roll_belt      : num  NA NA NA NA NA NA
##  $ skewness_roll_belt.1    : num  NA NA NA NA NA NA
##  $ skewness_yaw_belt       : logi  NA NA NA NA NA NA
##  $ max_roll_belt           : num  NA NA NA NA NA NA
##  $ max_picth_belt          : int  NA NA NA NA NA NA
##  $ max_yaw_belt            : num  NA NA NA NA NA NA
##  $ min_roll_belt           : num  NA NA NA NA NA NA
##  $ min_pitch_belt          : int  NA NA NA NA NA NA
##  $ min_yaw_belt            : num  NA NA NA NA NA NA
##  $ amplitude_roll_belt     : num  NA NA NA NA NA NA
##  $ amplitude_pitch_belt    : int  NA NA NA NA NA NA
##  $ amplitude_yaw_belt      : num  NA NA NA NA NA NA
##  $ var_total_accel_belt    : num  NA NA NA NA NA NA
##  $ avg_roll_belt           : num  NA NA NA NA NA NA
##  $ stddev_roll_belt        : num  NA NA NA NA NA NA
##  $ var_roll_belt           : num  NA NA NA NA NA NA
##  $ avg_pitch_belt          : num  NA NA NA NA NA NA
##  $ stddev_pitch_belt       : num  NA NA NA NA NA NA
##  $ var_pitch_belt          : num  NA NA NA NA NA NA
##  $ avg_yaw_belt            : num  NA NA NA NA NA NA
##  $ stddev_yaw_belt         : num  NA NA NA NA NA NA
##  $ var_yaw_belt            : num  NA NA NA NA NA NA
##  $ gyros_belt_x            : num  0 0.02 0 0.02 0.02 0.02
##  $ gyros_belt_y            : num  0 0 0 0 0.02 0
##  $ gyros_belt_z            : num  -0.02 -0.02 -0.02 -0.03 -0.02 -0.02
##  $ accel_belt_x            : int  -21 -22 -20 -22 -21 -21
##  $ accel_belt_y            : int  4 4 5 3 2 4
##  $ accel_belt_z            : int  22 22 23 21 24 21
##  $ magnet_belt_x           : int  -3 -7 -2 -6 -6 0
##  $ magnet_belt_y           : int  599 608 600 604 600 603
##  $ magnet_belt_z           : int  -313 -311 -305 -310 -302 -312
##  $ roll_arm                : num  -128 -128 -128 -128 -128 -128
##  $ pitch_arm               : num  22.5 22.5 22.5 22.1 22.1 22
##  $ yaw_arm                 : num  -161 -161 -161 -161 -161 -161
##  $ total_accel_arm         : int  34 34 34 34 34 34
##  $ var_accel_arm           : num  NA NA NA NA NA NA
##  $ avg_roll_arm            : num  NA NA NA NA NA NA
##  $ stddev_roll_arm         : num  NA NA NA NA NA NA
##  $ var_roll_arm            : num  NA NA NA NA NA NA
##  $ avg_pitch_arm           : num  NA NA NA NA NA NA
##  $ stddev_pitch_arm        : num  NA NA NA NA NA NA
##  $ var_pitch_arm           : num  NA NA NA NA NA NA
##  $ avg_yaw_arm             : num  NA NA NA NA NA NA
##  $ stddev_yaw_arm          : num  NA NA NA NA NA NA
##  $ var_yaw_arm             : num  NA NA NA NA NA NA
##  $ gyros_arm_x             : num  0 0.02 0.02 0.02 0 0.02
##  $ gyros_arm_y             : num  0 -0.02 -0.02 -0.03 -0.03 -0.03
##  $ gyros_arm_z             : num  -0.02 -0.02 -0.02 0.02 0 0
##  $ accel_arm_x             : int  -288 -290 -289 -289 -289 -289
##  $ accel_arm_y             : int  109 110 110 111 111 111
##  $ accel_arm_z             : int  -123 -125 -126 -123 -123 -122
##  $ magnet_arm_x            : int  -368 -369 -368 -372 -374 -369
##  $ magnet_arm_y            : int  337 337 344 344 337 342
##  $ magnet_arm_z            : int  516 513 513 512 506 513
##  $ kurtosis_roll_arm       : num  NA NA NA NA NA NA
##  $ kurtosis_picth_arm      : num  NA NA NA NA NA NA
##  $ kurtosis_yaw_arm        : num  NA NA NA NA NA NA
##  $ skewness_roll_arm       : num  NA NA NA NA NA NA
##  $ skewness_pitch_arm      : num  NA NA NA NA NA NA
##  $ skewness_yaw_arm        : num  NA NA NA NA NA NA
##  $ max_roll_arm            : num  NA NA NA NA NA NA
##  $ max_picth_arm           : num  NA NA NA NA NA NA
##  $ max_yaw_arm             : int  NA NA NA NA NA NA
##  $ min_roll_arm            : num  NA NA NA NA NA NA
##  $ min_pitch_arm           : num  NA NA NA NA NA NA
##  $ min_yaw_arm             : int  NA NA NA NA NA NA
##  $ amplitude_roll_arm      : num  NA NA NA NA NA NA
##  $ amplitude_pitch_arm     : num  NA NA NA NA NA NA
##  $ amplitude_yaw_arm       : int  NA NA NA NA NA NA
##  $ roll_dumbbell           : num  13.1 13.1 12.9 13.4 13.4 ...
##  $ pitch_dumbbell          : num  -70.5 -70.6 -70.3 -70.4 -70.4 ...
##  $ yaw_dumbbell            : num  -84.9 -84.7 -85.1 -84.9 -84.9 ...
##  $ kurtosis_roll_dumbbell  : num  NA NA NA NA NA NA
##  $ kurtosis_picth_dumbbell : num  NA NA NA NA NA NA
##  $ kurtosis_yaw_dumbbell   : logi  NA NA NA NA NA NA
##  $ skewness_roll_dumbbell  : num  NA NA NA NA NA NA
##  $ skewness_pitch_dumbbell : num  NA NA NA NA NA NA
##  $ skewness_yaw_dumbbell   : logi  NA NA NA NA NA NA
##  $ max_roll_dumbbell       : num  NA NA NA NA NA NA
##  $ max_picth_dumbbell      : num  NA NA NA NA NA NA
##  $ max_yaw_dumbbell        : num  NA NA NA NA NA NA
##  $ min_roll_dumbbell       : num  NA NA NA NA NA NA
##  $ min_pitch_dumbbell      : num  NA NA NA NA NA NA
##  $ min_yaw_dumbbell        : num  NA NA NA NA NA NA
##  $ amplitude_roll_dumbbell : num  NA NA NA NA NA NA
##   [list output truncated]
```

#### Data Cleaning
Looking at the structure of the first few rows of the train dataset, we will notice that there is a lot of "NA" and "#DIV/0!" values already present, hence we filter out those records. Furthermore, we made sure that these values are all converted to "NA" seeing the na.strings argument when reading the csv file for easier removal.


```r
cleantrain <- train[,colSums(is.na(train)) == 0]
cleantest <- test[,colSums(is.na(test)) == 0]
```

### Feature Selection
Let us reduce the number of variables or columns and take only what we need for the training set. 
Variables/Columns to remove: X (Index), Username, Timestamps, and Windows.

```r
cleantrain <- cleantrain[,-c(1:7)]
cleantest <- cleantest[,-c(1:7)]
dim(cleantrain)
```

```
## [1] 19622    53
```

```r
dim(cleantest)
```

```
## [1] 20 53
```

Take a look at the variable names to get a glimpse of the predictors that we will use in building the model. 

```r
colnames(cleantrain)
```

```
##  [1] "roll_belt"            "pitch_belt"           "yaw_belt"            
##  [4] "total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"        
##  [7] "gyros_belt_z"         "accel_belt_x"         "accel_belt_y"        
## [10] "accel_belt_z"         "magnet_belt_x"        "magnet_belt_y"       
## [13] "magnet_belt_z"        "roll_arm"             "pitch_arm"           
## [16] "yaw_arm"              "total_accel_arm"      "gyros_arm_x"         
## [19] "gyros_arm_y"          "gyros_arm_z"          "accel_arm_x"         
## [22] "accel_arm_y"          "accel_arm_z"          "magnet_arm_x"        
## [25] "magnet_arm_y"         "magnet_arm_z"         "roll_dumbbell"       
## [28] "pitch_dumbbell"       "yaw_dumbbell"         "total_accel_dumbbell"
## [31] "gyros_dumbbell_x"     "gyros_dumbbell_y"     "gyros_dumbbell_z"    
## [34] "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"    
## [37] "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"   
## [40] "roll_forearm"         "pitch_forearm"        "yaw_forearm"         
## [43] "total_accel_forearm"  "gyros_forearm_x"      "gyros_forearm_y"     
## [46] "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"     
## [49] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
## [52] "magnet_forearm_z"     "classe"
```

### Data Modeling - Random Forest
Why Random Forest?
One of the advantages of RF is it's good accuracy results. Additionally, it is a great algorithm for classification problems which suits our case. Also, it's default parameters already return good results and is good at avoiding overfitting. 

#### Partition the data into train and test/validation set

```r
#Here we will use the standard 60/40 data split 
set.seed(28)
split <- createDataPartition(train$classe, p = 0.6, list = FALSE)
modeltrain <- cleantrain[split,]
modeltest <- cleantrain[-split,]
dim(modeltrain); dim(modeltest)
```

```
## [1] 11776    53
```

```
## [1] 7846   53
```

#### Build the model with cross validation 

```r
RFModel <- train(classe ~ .
                , data = modeltrain
                , method = "rf"
                , metric = "Accuracy"  # Accuracy because we are expecting a categorical outcome variable
                , preProcess=c("center", "scale") # Normalize to at least improve accuracy
                , trControl=trainControl(method = "cv"
                                        , number = 4 # Folds of the training data
                                        , p= 0.60
                                        ))

print(RFModel, digits = 4)
```

```
## Random Forest 
## 
## 11776 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## Pre-processing: centered (52), scaled (52) 
## Resampling: Cross-Validated (4 fold) 
## Summary of sample sizes: 8831, 8833, 8832, 8832 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa 
##    2    0.9880    0.9849
##   27    0.9897    0.9870
##   52    0.9810    0.9759
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 27.
```

#### Predicting 

```r
predicting <- predict(RFModel, modeltest)
```

#### Model Evaluation
Let us now take a look at the statistical results of the model built. 

```r
confusionMatrix(predicting, modeltest$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2229   15    0    0    0
##          B    2 1497    6    0    2
##          C    1    6 1350   14    4
##          D    0    0   12 1270    2
##          E    0    0    0    2 1434
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9916          
##                  95% CI : (0.9893, 0.9935)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9894          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9987   0.9862   0.9868   0.9876   0.9945
## Specificity            0.9973   0.9984   0.9961   0.9979   0.9997
## Pos Pred Value         0.9933   0.9934   0.9818   0.9891   0.9986
## Neg Pred Value         0.9995   0.9967   0.9972   0.9976   0.9988
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2841   0.1908   0.1721   0.1619   0.1828
## Detection Prevalence   0.2860   0.1921   0.1752   0.1637   0.1830
## Balanced Accuracy      0.9980   0.9923   0.9915   0.9927   0.9971
```

#### Expected Out of Sample Error 

```r
ooserror <- 1 - as.numeric(confusionMatrix(modeltest$classe, predicting)$overall[1])
```

**Accuracy: 99.16%**

**Expected Out of Sample Error: 0.0084 or 0.84%**

#### Final model and Top 20 significant variables in our model

```r
RFModel$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.71%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3338    8    0    0    2 0.002986858
## B   17 2257    5    0    0 0.009653357
## C    0   10 2037    7    0 0.008276534
## D    0    0   24 1906    0 0.012435233
## E    0    2    5    4 2154 0.005080831
```

```r
varImp(RFModel)
```

```
## rf variable importance
## 
##   only 20 most important variables shown (out of 52)
## 
##                      Overall
## roll_belt            100.000
## pitch_forearm         57.542
## yaw_belt              52.069
## roll_forearm          45.670
## magnet_dumbbell_z     43.871
## pitch_belt            43.705
## magnet_dumbbell_y     41.917
## accel_dumbbell_y      23.112
## accel_forearm_x       17.268
## roll_dumbbell         17.180
## magnet_dumbbell_x     16.488
## magnet_belt_z         15.420
## accel_dumbbell_z      14.033
## magnet_belt_y         13.601
## total_accel_dumbbell  13.087
## magnet_forearm_z      12.838
## accel_belt_z          12.493
## magnet_belt_x         10.220
## gyros_belt_z          10.145
## roll_arm               9.822
```

### Model Validation 

```r
print(predict(RFModel, newdata = cleantest))
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

*The model built is still up for the Course Project Prediction Quiz
