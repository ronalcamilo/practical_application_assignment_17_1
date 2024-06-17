# Practical Application III: Comparing Classifiers

### Overview
In this practical application, th objective is to compare the performance of the classifiers we encountered in this section, namely K Nearest Neighbor, Logistic Regression, Decision Trees, and Support Vector Machines. We will utilize a dataset related to marketing bank products over the telephone.

### Data:
The data can be found in this link: 

[bank-additional-full.csv](https://github.com/ronalcamilo/practical_application_assignment_17_1/blob/main/data/bank-additional-full.csv)

Notebook Link
The following notebook contains all the development of the analysis carried out.

[prompt_III.ipynb](https://github.com/ronalcamilo/practical_application_assignment_17_1/blob/main/prompt_III.ipynb)

### Business Understanding
To gain a better understanding of the data, please read the information provided in the UCI link above, and examine the Materials and Methods section of the paper. How many marketing campaigns does this data represent?

This data represents 17 campaigns taken from May 2008 to November 2010.

### Read in the Data
![image](https://github.com/ronalcamilo/practical_application_assignment_17_1/blob/main/images/info.png)

### Understanding the Features

Input variables:
* bank client data:
1 - age (numeric)
2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
5 - default: has credit in default? (categorical: 'no','yes','unknown')
6 - housing: has housing loan? (categorical: 'no','yes','unknown')
7 - loan: has personal loan? (categorical: 'no','yes','unknown')
# related with the last contact of the current campaign:
8 - contact: contact communication type (categorical: 'cellular','telephone')
9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
* other attributes:
12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
14 - previous: number of contacts performed before this campaign and for this client (numeric)
15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
* social and economic context attributes
16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
17 - cons.price.idx: consumer price index - monthly indicator (numeric)
18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)
19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
20 - nr.employed: number of employees - quarterly indicator (numeric)
* Output variable (desired target):
21 - y - has the client subscribed a term deposit? (binary: 'yes','no')

### Nulls
We searched for nulls within the data to understand if a strategy to autocomplete that data is necessary, but we found that the dataset does not contain nulls.
![image](https://github.com/ronalcamilo/practical_application_assignment_17_1/blob/main/images/nulls.png)

![image](https://github.com/ronalcamilo/practical_application_assignment_17_1/blob/main/images/nulls_chart.png)

### Understanding the Task
#### analysis
once we took the data we reviewed what is the proportion of information for the column and in which we have an acceptance of 88.73 (36548) Not subscribed and 11.27 (4640) Subscribed.

![image](https://github.com/ronalcamilo/practical_application_assignment_17_1/blob/main/images/portion.png)

correlation between characteristics
![image](https://github.com/ronalcamilo/practical_application_assignment_17_1/blob/main/images/correlations.png)

After analyzing the correlations we found properties with high correlations which we proceeded to remove in order to simplify the model and to have a higher accuracy
we will only use:
* age
* job
* marital
* education
* default
* housing 
* loan
* contact 
* subscribed

#### Uniques values
![image](https://github.com/ronalcamilo/practical_application_assignment_17_1/blob/main/images/unique.png)

### Train/Test Split
Next, I split the data

```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42) 
```

###  Baseline model
using DummyClassifier we define a baseline
![image](https://github.com/ronalcamilo/practical_application_assignment_17_1/blob/main/images/baseline.png)

###  A Simple Model
After that we create our first classifier model, this is a LogisticRegression classifier which will train the X_train data and see if the performance is better than the baseline performance, we will run it with all our features and using the default parameters for the model.

### Score the Model

What is the accuracy of your model?

- Accuracy score for the logistic regressor for train is 90.96%
- Accuracy score for the logistic regressor for test is 90.87%

Using the acuracy_score function we can see that the score we get is better than our baseline model.

### Model Comparisons

Using the default parameters, we ran all the classifiers models that we have seen and this are the results:

![](./images/default_params_models.png)

- SVC is the classifier model that took the longest to train the data and the fastest model was KNN
- The model with the highest accuracy on the test data was LogisticRegression
- All models obtained greater precision than the reference model.

### Improving the model
  
Now that we have some basic models on the board, we try to improve them. Following the next steps:

#### More feature engineering and exploration. For example, should we keep the gender feature? Why or why not?

To identify the influence of each feature linearly on the result, we make a plot of the correlations.

![](./images/CorrelationHeatMap.png)

Here we can identify that there are some features that are highly correlated with the resulting variable such as: ['duration', 'pdays', 'previous', 'emp.var.rate', 'euribor3m', 'nr.employed', 'poutcome'] so I decided to work with these features.


#### Hyperparameter tuning and grid search

For this test I will use the params that I got as the most correlated with the target feature. I ran the model using GridSearchCV with their repective hyperparamerers for each model and using cross validation to get the best estimator. The next picture is showing the performance

![](./images/grid_search.png)

The results show that the best test score was obtained by the Decision Tree model and the worst test score was obtained by LogisticRegression, but it is still good, it is above 90%.

The SVC model had a very low performance in terms of time, since it took more than 2 hours to train the model, this is a big difference compared to the rest of the models.

#### Adjust your performance metric

In this Step my Objective was to add new scores to the table and also visualize the Confusion Matrix for each model to get a better understanding. For instance the following plot shows the Confusion Matrix for KNN.

![](./images/KNN_Confusion_matrix.png)

I also added the following scores, Precision, Recall and F1 since the scores we were dealing with until now were precision scores.

![](./images/performance_metrics.png)

The results show us that Decision Tree obtained the best results in all metrics and also a reasonable training time. For this reason we can conclude that it is the best model in this case.


### Findings

We analyzed different classification methods such as KNN, Logistic Regression, Decision Tree and SVM with different parameters, in order to find the method that had the best performance for this project, in the process we found that Decision Tree obtained the best results in all metrics and also a reasonable training time. For this reason we can conclude that it is the best model for this dataset.

The SVC did not obtain bad results in terms of score, it could even be said that it was the second best, but the long time it took to train leads us to think that this may be a definitive factor to discard it and opt for Decision Tree.

Identifying some important factors can help us be more efficient in future campaigns, such as:

- Duration is the aspect that is most positively correlated with success,
- The outcome of the previous marketing campaign is also a very important feature
- success is most likely to occur in the last month of each quarter (March, June, September and December)

### Next Steps and Recomendations

- The next campaigns can be based on the aspects already identified in this analysis, trying to focus the maximum effort on the last months of each quarter, clients with whom there has already been success in previous campaigns and also trying to generate sufficient interest in the client. so that calls last longer.

- The models could be tested with some other features that were not used in this study.
