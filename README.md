# RainfallPrediction
Rainfall prediction in Australia using Machine learning 


**1. Introduction**

Weather is a significant external phenomenon that can not be controlled and affects our lives in various ways. From the early days of the cavemen era, predicting weather has been deemed as one of the biggest mysteries to solve since it aided in farming, water supply, safety, and increased success in gathering food necessary for survival. In the current times, aside from helping us to determine how to dress the next morning, many businesses and industries are greatly dependent on weather to prolong their survival. Farmers and gardeners plan for crop irrigation and protection, construction companies can better understand when to pour concrete before or ahead of a heavy rain, amusement parks need to take action on how to serve visitors during rainy or snowy weather, among others. Though the result can not be averted, we can try to predict weather more accurately with the latest technological advancement.

Australia's weather and climate are changing in response to global warming. Previous events such as the bushfire in 2019-2020 captured the world’s attention on the nation’s extreme weather conditions due to human interference. Because of its vast land mass and climate difference, Australia’s weather can vary significantly from location to location. The goal of this project is to predict, using the dataset “Rain in Australia” from Kaggle, whether there will be rain next day in Australia. We opted to use a mix of classification models and logistic regression on a target variable (RainTommorow) using supervised learning models. Our project uses several data mining models including sklearn decision tree (benchmark), boosted decision trees: XGBoost and LightGBM decision tree, and logistic regression. to learn the patterns of Australia’s weather and performance evaluation metrics such as precision, recall, and AUC. 


**2. Data Understanding**

Our dataset is sourced from Kaggle: https://www.kaggle.com/jsphyg/weather-dataset-rattle-package
The dataset contains 23 columns and 145,460 entries divided into 49 locations with each location having around 3,000 entries. Summary of important attributes are as follows:
Feature
Types
Missing Values
Meaning
Evaporation
Numerical
~ 43% 
Class A pan evaporation (mm) in the 24 hours to 9am
Sunshine
Numerical
~ 48%
The number of hours of bright sunshine in the day.
WindDir9am/
WindDir3pm
Categorical 
(Nominal)
~ 7%  
~ 3% 
Direction of strongest gust in the past 24 hours to midnight. Wind direction is important because wind carries weather to a location. 
WindSpeed9am/
WindSpeed3pm
Numerical
~ 3% 
Wind speed (km/hr) 10 minutes prior to 9am and prior to 3pm. WindSpeed is an indication of how fast moisture is being transported
Humidity9am/ Humidity3pm
Numerical
~ 1.8%
~ 3%
Humidity % at 9am and 3pm
Cloud3pm/
Cloud9am
Numerical
~40% 
Fractions of sky obscured by clouds at 9am and 3pm. 0 = clear sky, 8 = completely overcast.
Pressure9am/
Pressure3pm


Numerical
~ 10%
~ 10%
Atmospheric pressure (hpa) reduced to mean sea level at 9am and 3pm.
RainToday
Categorical 
(Nominal)
~ 2% 
Measured by precipitation in mm. Value is 1 if precipitation in 24 hours to 9am exceeds 1mm, otherwise value is 0. 
RainTomorrow (Label)
Categorical 
(Nominal)
~ 2% 
Value is 1 if tomorrow’s precipitation in 24 hours to 9am exceeds 1mm, otherwise value is 0.



The entries are indexed by dates as early as October 2007 up to May 2017. However, we found severe missing entries (indicated in purple) in 2007 and 2008 in most locations, thus we decided to drop all entries before January 1, 2019.

One of the major issues was that a few columns have several missing values. Sunshine has 48% missing values, Evaporation 43%, Sunshine 48%, Cloud9am 38%, Cloud3pm 40%, and are generally location-specific such as in Newcastle, Mount Ginini, Salmon Gums, due to the locations not having such sensors. We decided not to drop these attributes, as it is highly correlated with label RainTomorrow compared to other features as seen in the correlation heatmap, such as Sunshine has negative 0.45 correlation, Cloud3pm has 0.38 correlation, and Evaporation negative 0.12 correlation. 

**3. Data Preprocessing**

1) Filling in null values
There are two approaches in filling in the null values, with the first one being much more simple. However,  we will later show that the second approach will lead to better model performance.

1. First approach: Fill null data with mean or mode of that column

A simple method is to fill in the null data of a column using its mean if it is a numerical feature, or fill using its mode if it is a categorical feature.

2.Second approach: Fill null data with 3 closest locations to target on the same date

As weather is location-specific, it is usually similar to neighbouring locations to a certain extent. To deal with missing values, we decided to fill in target values with the closest neighbour’s value on the same date. First, we identify each location’s coordinates using Geopy Geocoders library and return a dataframe of city name, latitude, and longitude. A problem we encountered was the location ‘Goldcoast, Australia’ was capturing a different location also called ‘Goldcoast’ in San Diego, California, thus we manually inserted the correct coordinates for the target location.

2.1.   Fill null using a location’s 3 closest locations
Next, we calculated the distances of every possible location combination, for instance, Newcastle to Williamtown is 13.02 km apart, and so on. To fill in the target location’s missing values, we searched for the target’s first closest neighbour by the same date if they had the target’s missing values, and filled in if available for that date. If not, move on to the target's second closest neighbour and so on, until the third closest neighbour. For this we used the geoencoder package from Python and “Google Maps” as the agent to compute the latitude and longitude of the locations, and then used the Haversine formula to arrive at the distance:


2.2.    Fill null with yesterday’s data
Some locations still have missing values due to its neighbours also having null values (e.g. Canberra and Mount Ginini share the same number of null values). Our idea of tackling this problem by filling in with yesterday’s data, as yesterday’s weather is the best predictor we have for predicting tomorrow’s weather. Note that Uluru, Katherine, and Nhil’s entries start from 2013, much later than other locations, and these three locations do not have null values (see table Data Loss on Missing Date Entry). The final heatmap table after data preprocessing can be seen on the right. 

2) Min-max normalization
The dataset contains 16 numerical features (eg. MinTemp, MaxTemp, Rainfall) in total. Since they are weather measurements that have different scales, we apply min-max scaling to all numerical features such that every value is between values of 0 to 1. Z-score normalization is not chosen because the numerical features do not necessarily follow a normal distribution.

3) Categorical feature encoding
Since RainToday and RainTomorrow have ‘object’ as data type, we applied One-hot encoding by creating dummy variables and dropping the original column (1 for Yes, 0 for No). There are features such as ‘WindGustDir ', ‘WindDir9am’ that contains 16 different values of wind direction (north (N), north-northeast (NNE), northeast (NE), east-northeast (ENE), etc). Instead of creating 15 more columns for each feature, we implement categorical encoding according to each number’s presence in a scale out of 3 to capture the dimension difference. 

**4. Model Building**

After dropping, the dataset has 22% positive labels (rain tomorrow) and 78% negative labels (no rain). Our benchmark for this project is 78% accuracy, as the majority classifier will get a 78% result if no data preprocessing and modeling is done. 

After finishing the preprocessing, we had our final CSV which contained normalized, complete columns. However, we had taken 2 different approaches to transforming the dataset:

Filling the missing values with Mean/Mode of the column vs Filling the missing value with nearest 3 locations (Approach 1- Approach 2): Since there are different locations (total of 49 unique values), our first approach was to fill the missing values with mean and the other by using closest 3 locations to do the same, as mentioned in the data-preprocessing.

Splitting the train-testing data using 80-20 split as a whole vs Splitting the train-testing data using 80-20 split of each location: Since, there are 49 unique locations in the dataset, with some locations having more entries than other locations, we believed that a simple train test split would lead to more representation of certain locations over others. Hence, we ran an algorithm that would select 80% of data from each location, and add that to the training set, and the remaining 20% would go into the testing set.

We then arrived at the 4 models that we believed would be appropriate to predict the rain. The models that we decided to use were:

Sklearn decision tree
Decision Tree Classifier is one of the simplest and most understandable machine learning models and can give insightful explanations because of the model’s simple logic. In our project of predicting rain in Australia, we think implementing decision tree classifiers is appropriate given the scope of our class. The first decision tree classifier we use is sklearn decision tree, a basic decision tree and will be our ‘benchmark’ for our other tree models (xgboost and lightgbm).

LightGBM (Boosted Decision Tree)
LightGBM or Light Gradient Boosting is a tree-based algorithm that is widely used because of its accuracy, efficiency, and stability. It is based on a few key principles. Firstly, decision trees are weak learners and hence a single decision tree usually gives a poor performance. Hence, in a gradient boosted decision tree, trees are built sequentially where the first tree learns to fit the target variable (in our case RainTomorrow), then the second tree learns to fit the residual/error of the first tree, then the third tree learns to do the same for the second tree, and so on so forth. Further, LightGBM grows leaf-wise, while other algorithms grow level-wise

XGBoost (Boosted Decision Tree)
Similar to LightGBM, XGBoost is a library that also provides gradient boosted decision trees that have similar working principles. The main difference between the two libraries is that XGBoost grows level-wise while LightGBM grows leaf-wise. For instance, using the diagram in the above, XGBoost decision tree can only expand the 3rd level after the 2nd level was fully expanded, while LightGBM decision tree can expand to 4th level before the 3rd level finishes expanding. To avoid overfitting, the maximum depth of the tree is limited to no more than 2 levels.


Logistic Regression
Logistic Regression is one of the basic and popular algorithms to solve a classification problem. Logistic regression is a method which is used to predict a dependent variable given a set of independent variables, such that the dependent variable is categorical. 
The assumptions of the Logistic Regression model are:
The dependent variable should be categorical variable (nominal or ordinal)
The dataset is free from outliers.
The data in the dataset is free from multicollinearity (high correlation)

Logistic Regression estimates the log odds of the event, the regression function is defined as:
 
where betas are the coefficients of the independent variable respectively.
  
Logistic regression is highly prone to overfitting problems. This is because by including more independent variables to the logistic regression model is highly likely to increase the accuracy. However, the improvement in the accuracy is at cost of reducing the generalizability of the model, and thus overfitting is resulted. Hence, in our model, the independent variables with feature importance less than 0.5 are dropped with not significantly affect the accuracy. 


**5. Performance Evaluation**

To evaluate whether our data preprocessing efforts improve the model, we splitted the dataset into training and testing sets and we evaluate the model performance using the testing set. There are 4 cases we tried in each model in terms of how to fill in null values and how training and testing set are split:

Case 1: Fill null values with mean + split training and testing set randomly using train_test_split
Case 2: Fill null values with mean + split training and testing set by location 
Case3: Fill in null values with 3 closest neighbours' values + split training and testing set randomly using train_test_split
Case 4: Fill in null values with 3 closest neighbours' values + split training and testing set by location

1) Sklearn Decision Tree

(i) Accuracy, Precision, and Recall 
The decision tree model’s accuracy was highest in case 1, which is filled by mean and random split, at 84%, followed very closely by case 3 and case 4. However, we can not simply pick the model with the highest accuracy. Precision and recall give a better insight on the confusion matrix. Taking the highest numbers, case 4 gives macro average precision at 0.79 and recall at 0.71, yet case X slightly exceeds the recall at 0.81 but a slightly lower precision at 0.70. 


(ii) ROC curve and AUC
For case 4, the ROC curve provides an area under the curve of 0.8399, the lowest of all models we tried. 


2) LightGBM

(i) Accuracy, Precision, and Recall 	
The LightGBM model achieved the highest accuracy of 87% when we applied the case-4 criteria, where we filled the missing values by the closest location, and split the data into training and testing by location. For the positive label (Rain_tomorrow= 1, which means Yes) the precision was 76% whereas the recall was 59%. For the negative label (Rain_tomorrow=0, which means No) the precision was 89% and recall was 95%. It is noticeable that the model does have low precision and recall in the positive class, however since the model ha several locations with a diverse weather condition leading to rain, achieving an accuracy of 89% beats the benchmark of 77% (with majority classifier labelling all entries as 0), and Sklearn decision tree with 84% accuracy

(ii) ROC curve and AUC
For the case-4, the ROC curve had an area under the curve of 0.885, which exceeds other models. Since an ideal classifier would be at the top left of the ROC curve, lightGBM model performs better than the benchmark.

3) XGBoost

(i) Accuracy, Precision, and Recall 
The model achieves the highest accuracy of 84.9% in case 3, which is filling null with 3 closest neighbours’ values and splitting training and testing sets randomly using train_test_split. For the negative label (i.e., RainTomorrow = 0), the precision was 87% and the recall was 95%. For the positive label (RainTomorrow = 1), the precision was 73% and the recall was 51%. Compared to LightGBM case 4’s results, this model is less performant.

The model’s accuracy, precision and recall are generally higher when we fill null values with 3 closest neighbours’ values compared to filling with mean or mode of the column. However, the model’s performance does not improve much when we choose to split training and testing data set by location instead of splitting by random with train_test_split.

(ii) ROC curve and AUC
For case 3, the ROC curve has an area under the curve of 0.886, which is very close to LightGBM model’s performance. It also curves towards the top left corner, which is ideal.
4) Logistic Regression
(i) Accuracy, Precision, and Recall 
The model achieves the highest accuracy of 84.9% in case 3, which is filling null with 3 closest neighbours’ values and splitting training and testing sets randomly using train_test_split. For the negative label (i.e., RainTomorrow = 0), the precision was 87% and the recall was 94%. For the positive label (RainTomorrow = 1), the precision was 73% and the recall was 53%. Compared to LightGBM case 4’s results, this model is less performant.

The model’s accuracy, precision and recall are generally higher when we fill null values with 3 closest neighbours’ values compared to filling with mean or mode of the column. However, the model’s performance does not improve much when we choose to split training and testing data set by location instead of splitting by random with train_test_split.

(ii) ROC curve and AUC
For case 3, the ROC curve has an area under the curve of 0.873, which is very close to LightGBM model’s performance. It also curves towards the top left corner, which is ideal.


**Summary of Results**

In summary, case 4 from LightGBM boosted decision tree model has the best performance:

Prediction Measures
Sklearn DT
XGBoost DT
LightGBM DT
Logistic Regression
Case
1
2
3
4
1
2
3
4
1
2
3
4
1
2
3
4
Accuracy
0.84
0.83
0.84
0.84
0.85
0.84
0.85
0.85
0.86
0.85
0.86
0.87
0.84
0.84
0.85
0.85
Precision^
0.79
0.79
0.80
0.79
0.79
0.79
0.80
0.80
0.81
0.81
0.82
0.83
0.79
0.79
0.80
0.80
Recall^
0.68
0.67
0.70
0.71
0.71
0.71
0.73
0.72
0.73
0.74
0.76
0.77
0.71
0.71
0.73
0.73

^ Using the macro average (“macro avg”) in the classification report generated by sklearn’s classification_report()
# For the full classification report for all the above cases, see Appendix item 2

6. Hyperparameter Tuning

Sklearn Decision Tree
To avoid overfitting, we implemented hyperparameters to stop the tree from growing, such as max_depth indicating tree’s maximum number of branches and max_leaf_nodes to grow tree with most informative nodes first. Since we do not know what is the best number for the hyperparameters, we decided to use GridSearchCV which will run cross validation within our dataset and found max_depth = 5 and max_leaf_nodes = 100 is the best hyperparameter. 

LightGBM 
For the LightGBM model there are 3 main parameters that can be tuned for better results, according to the LightGBM documentation:
num_leaves:  This is the main parameter to control the complexity of the tree model. Theoretically, we can set num_leaves = 2^(max_depth) to obtain the same number of leaves as depth-wise trees. However, this simple conversion is not good in practice. Hence setting the num_leaves to 127 when max_depth= 7, may lead to overfitting. The model we used had a max_depth= 5, and num_leaves= 31 (which is less than 2^5=32)
min_data_in_leaf: This is an important parameter to prevent over-fitting in a leaf-wise tree. Its optimal value depends on the number of training samples and num_leaves. Setting it to a large value can avoid growing too deep a tree, but may cause under-fitting. In practice, setting it to tens or hundreds. We selected the default value of 20
max_depth= was set to 5 to prevent overfitting

XGBoost
There are 3 main parameters that can be tuned:
eta:  This parameter is the step size shrinkage used in update to prevent overfitting. After each boosting step, we can directly get the weights of new features, and eta shrinks the feature weights to make the boosting process more conservative
max_depth:  This is an important parameter to prevent overfitting as it controls the maximum depth of the decision tree. Since it’s recommended to use a simple base model in ensemble learning, max_depth should be kept at a low value
gamma:   This parameter controls how conservative the decision tree algorithm is, since it is the minimum loss reduction required to make a further partition on a leaf node of the tree. The higher the value, the more conservative the model is

Logistic Regression
The 2 main parameters that we tuned were:
Solver: we selected the ‘newton-cg’ solver since it performed the best out of the other methods which included, liblinear’, ‘lbfgs’, ‘sag’, and ‘saga’ after gridsearchcv
Penalty: We used L2 for the penalty, after deploying grid search cv.

**7. Further Improvements**

In this research, the model was formulated for predicting rain probability in generalized climatology. However, different locations have their unique microclimate system, which might affect the rain process. Hence, there should be a locational dependence for the rain probability. For the future improvement, we can do clustering for the location with similar patterns in the climatology dataset, and build different models based on the microclimate clusters. We did run a KMeans cluster for our models to check whether we could improve the accuracy, however since the dataset is fairly limited (with only 23 columns) and several other factors need to be taken into account for microclimatic clusters, the clustering did not improve the accuracy of our model. 

Besides, Deep Learning can provide a new opportunity for us to improve the model. Deep Learning models can better accommodate the noise and bias in the dataset. In addition, the network structure can enable to catch patterns on both linearity and non-linearity (relu activation function). It is believed that the rain prediction model can be improved through adopting the Deep learning model.

8. Conclusion
Machine learning is one of the technological advancements to better understand the patterns and attribute’s impact on the label. In this project, we used four different models to try and capture the problem from various angles. From the perspective of accuracy, precision, recall, LightGBM model provides the best prediction on whether it will rain or not in 49 locations across Australia. We used the Sklearn Decision Tree as the benchmark for the boosted decision tree models and logistic regression was used to further explore the results. Since the LightGBM model presented the most promising results, we plotted a feature importance graph to see if we could derive any insights.

As can be seen from the curve, the Pressure at 3pm, Humidity at 3pm, Pressure9am, WindGustSpeed and Sunshine, are the top 5 most important features in determining whether it will rain tomorrow or not. These factors could be analyzed more seriously for future prediction of rain. 

