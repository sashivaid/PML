---
title: "Practical Machine Learning Course Project"
author: "SV"
date: "March 11, 2018"
output: 
  html_document: 
    keep_md: yes
---



## Overview

Human activity recognition has emerged as a key research area in identifying human activities and in quantifying how well the activity is performed. It is now possible to collect large data sets of human activity measurements quite inexpensively. One such data set is the Weight Lifting Exercise data set.

The goal for this project is to train a model based on a training set with appropriate features and then predict the class of the weight lifting exercise (class A indicating correct executing and class B - E indicating improper execution).



## Exploratory Analysis 

Initial analysis of the data shows that there are 67 features in the training set that do not contain any value for more than 95% of the rows.

## Feature Selection for Model fitting
Let us identify variables with very little variability since these are not good predictors.


Results of the "nearZeroVar"" function indicate that there are 60 variables that are not good predictors. So , we will remove these from the training data set.



Continued analysis of the data shows that there are still some computed variables like standard deviation, variance, mean, min, max etc that can be omitted from training the model. These can be computed from the raw data and hence not useful for model training. So, we will remove them.




The resulting data has 59 features. Out of this, the first 6 features describe the row number, the identification of the person who did the exercise, the date/time stamps when the exercise was performed and the sliding window number form which the measurements were extracted. These do not have any bearing on the actual exercise and the measurements generated when performing the exercise. So, we will remove these features also before training the model.



## Cross validation Data Set
For the purposes of cross validating our model, we will randomly select 25% of the training data as a hold out set to perform cross validation.



There are now 14718 rows to train the model and 4904 rows to perform cross validation.

## Model Fitting
Now, we will attempt to fit a random forest model using the training set (after removing unwanted features and after removing the rows for the cross validation). The seed has been set to 12321 to ensure reproducibility.




```r
set.seed(12321)
x <- training_final[,-53]
y <- training_final[,53]
fitControl <- trainControl(method = "cv",
                           number = 5,
                           allowParallel = TRUE)

fit <- train(x,y, method="rf",data=training_final,trControl = fitControl)
```



## Accuracy

The **model accuracy** based on the training data is as follows:

```
## Random Forest 
## 
## 14718 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 11774, 11775, 11774, 11774, 11775 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9913712  0.9890843
##   27    0.9913712  0.9890844
##   52    0.9862072  0.9825520
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 2.
```
The **accuracy of each fold** as part of the cross validation within the training process is:

```
##    Accuracy     Kappa Resample
## 1 0.9911685 0.9888293    Fold1
## 2 0.9925246 0.9905433    Fold2
## 3 0.9915053 0.9892537    Fold5
## 4 0.9915082 0.9892575    Fold4
## 5 0.9901495 0.9875378    Fold3
```

## Out of sample error

The confusion matrix when predicting against cross validation set is

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1394    0    0    0    1
##          B    3  942    4    0    0
##          C    0    3  852    0    0
##          D    0    0   20  784    0
##          E    0    0    0    0  901
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9937         
##                  95% CI : (0.991, 0.9957)
##     No Information Rate : 0.2849         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.992          
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9979   0.9968   0.9726   1.0000   0.9989
## Specificity            0.9997   0.9982   0.9993   0.9951   1.0000
## Pos Pred Value         0.9993   0.9926   0.9965   0.9751   1.0000
## Neg Pred Value         0.9991   0.9992   0.9941   1.0000   0.9998
## Prevalence             0.2849   0.1927   0.1786   0.1599   0.1839
## Detection Rate         0.2843   0.1921   0.1737   0.1599   0.1837
## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
## Balanced Accuracy      0.9988   0.9975   0.9859   0.9976   0.9994
```
The **Out of sample error rate** when predicting on the cross validation data set is 0.0063214

## Plots

The plot of randomly selected predictors vs accuracy (using cross validation) is provided below
![](index_files/figure-html/unnamed-chunk-13-1.png)<!-- -->

Importance of the various features in the model:
![](index_files/figure-html/unnamed-chunk-14-1.png)<!-- -->

