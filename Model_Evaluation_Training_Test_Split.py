import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from Model_Evaluation_Plot_Functions import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df = pd.read_csv("module_5_auto.csv", header=0)

#Consider only the columns with numeric data for the analysis
df=df._get_numeric_data()
#drop the two unwanted columns
df.drop(['Unnamed: 0.1', 'Unnamed: 0'], axis=1, inplace=True)

y_data = df['price'] #Price is for Y
x_data=df.drop('price',axis=1) #everything except Price is fot X

############################ Train, Test data split and R^2 calculation #########################

#Split the df for Training and Test data
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.10, random_state=1)
print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])

lre = LinearRegression()
lre.fit(x_train[['horsepower']],y_train) #use the train set to test the model
print('R^2 of test set: ', lre.score(x_test[['horsepower']],y_test)) #calculate R^2 of test set
print('R^2 of train set: ', lre.score(x_train[['horsepower']],y_train)) #calculate R^2 of train set

#Exercise
x_train1,x_test1,y_train1,y_test1 = train_test_split(x_data,y_data,random_state=0, test_size=0.4)
print("number of test samples :", x_test1.shape[0])
print("number of training samples:",x_train1.shape[0])


#Use the exercise set to find the R^2
lre.fit(x_train1[['horsepower']],y_train1) #use the train set to test the model
print('R^2 of test set: ', lre.score(x_test1[['horsepower']],y_test1)) #calculate R^2 of test set
print('R^2 of train set: ', lre.score(x_train1[['horsepower']],y_train1)) #calculate R^2 of train set

######################## CROSS VALIDATION SCORE ############################################
# Sometimes you do not have sufficient testing data; as a result, you may want to perform cross-validation.
from sklearn.model_selection import cross_val_score

#This will partion (number of folds) the data intp 4 parts (cv=4), find the average R^2 of each fold and assign it to DF Rcross
Rcross = cross_val_score(lre, x_data[['horsepower']], y_data, cv=4) 
print('Average R^2 of each folds are: ', Rcross)
print("The mean of the folds are", Rcross.mean(), "and the standard deviation is" , Rcross.std())

"""
You can also use the function 'cross_val_predict' to predict the output. The function splits up the data into the specified number of folds, with one fold for testing and the other folds are used for training.
"""
from sklearn.model_selection import cross_val_predict
yhat = cross_val_predict(lre,x_data[['horsepower']], y_data,cv=4)
print(yhat[0:5])

#Exercise
Rc = cross_val_score(lre, x_data[['horsepower']], y_data, cv=2)
print('Average R^2 of each folds are: ', Rc)
print("The mean of the folds are", Rc.mean(), "and the standard deviation is" , Rc.std())


########################## Overfitting, Underfitting and Model Selection ############################

"""
It turns out that the test data, sometimes referred to as the "out of sample data", is a much better measure of how well your model performs in the real world. One reason for this is overfitting.
"""
# create Multiple Linear Regression objects and train the model using 'horsepower', 'curb-weight', 'engine-size' and 'highway-mpg' as features.
lr = LinearRegression()
lr.fit(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_train)
yhat_train = lr.predict(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
print('Few predictions using train set are: ', yhat_train[0:5])

yhat_test = lr.predict(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
print('Few predictions using test set are: ', yhat_test[0:5])

#Let's visualize the training data set predictions
Title = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution'
DistributionPlot(y_train, yhat_train, "Actual Values (Train)", "Predicted Values (Train)", Title)

#data visualization using test data set
Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
DistributionPlot(y_test,yhat_test,"Actual Values (Test)","Predicted Values (Test)",Title)

"""
Overfitting: Overfitting occurs when the model fits the noise, but not the underlying process. Therefore, when testing your model using the test set, your model does not perform as well since it is modelling noise, not the underlying process that generated the relationship. 
"""
from sklearn.preprocessing import PolynomialFeatures

#Let's create a degree 5 polynomial model.
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.45, random_state=0)
pr = PolynomialFeatures(degree=5)
x_train_pr = pr.fit_transform(x_train[['horsepower']])
x_test_pr = pr.fit_transform(x_test[['horsepower']])

poly = LinearRegression()
poly.fit(x_train_pr, y_train)
yhat = poly.predict(x_test_pr)
print("Predicted values:", yhat[0:4])
print("True values:", y_test[0:4].values)

#Visualize the data
PollyPlot(x_train['horsepower'], x_test['horsepower'], y_train, y_test, poly,pr)

print('R^2 of thte train data set: ', poly.score(x_train_pr, y_train))
print('R^2 of the test data set: ', poly.score(x_test_pr, y_test))

"""
At this point R^2 of test data is too low hence the model is worse performing.
Let's see how the R^2 changes on the test data for different order polynomials and then plot the results:
"""
Rsqu_test = []

order = [1, 2, 3, 4]
for n in order:
    pr = PolynomialFeatures(degree=n)
    
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    
    x_test_pr = pr.fit_transform(x_test[['horsepower']])    
    
    lr.fit(x_train_pr, y_train)
    
    Rsqu_test.append(lr.score(x_test_pr, y_test))

plt.plot(order, Rsqu_test)
plt.xlabel('order')
plt.ylabel('R^2')
plt.title('R^2 Using Test Data')
plt.text(3, 0.75, 'Maximum R^2 ')
plt.show()

# Exercise
pr1=PolynomialFeatures(degree=2)
x_train_pr1=pr1.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
x_test_pr1=pr1.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
poly1=LinearRegression().fit(x_train_pr1,y_train)
yhat_test1=poly1.predict(x_test_pr1)
Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
DistributionPlot(y_test, yhat_test1, "Actual Values (Test)", "Predicted Values (Test)", Title)

##### Ridge Regression ###########
#will review Ridge Regression and see how the parameter alpha changes the model.
pr=PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])
x_test_pr=pr.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])
from sklearn.linear_model import Ridge
RigeModel=Ridge(alpha=1)
RigeModel.fit(x_train_pr, y_train)
yhat = RigeModel.predict(x_test_pr)
print('Ridge Regression predicted:', yhat[0:4])
print('test set :', y_test[0:4].values)
"""
We select the value of alpha that minimizes the test error. To do so, we can use a for loop. We have also created a progress bar to see how many iterations we have completed so far.
"""
from tqdm import tqdm

Rsqu_test = []
Rsqu_train = []
dummy1 = []
Alpha = 10 * np.array(range(0,1000))
pbar = tqdm(Alpha)

for alpha in pbar:
    RigeModel = Ridge(alpha=alpha) 
    RigeModel.fit(x_train_pr, y_train)
    test_score, train_score = RigeModel.score(x_test_pr, y_test), RigeModel.score(x_train_pr, y_train)
    
    pbar.set_postfix({"Test Score": test_score, "Train Score": train_score})

    Rsqu_test.append(test_score)
    Rsqu_train.append(train_score)

#Plot the R^2 for different Alpha values
width = 12
height = 10
plt.figure(figsize=(width, height))

plt.plot(Alpha,Rsqu_test, label='validation data  ')
plt.plot(Alpha,Rsqu_train, 'r', label='training Data ')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.legend()

################## Grid Search ##############
"""
The term alpha is a hyperparameter. Sklearn has the class GridSearchCV to make the process of finding the best hyperparameter simpler.
"""
from sklearn.model_selection import GridSearchCV
parameters1= [{'alpha': [0.001,0.1,1, 10, 100, 1000, 10000, 100000, 100000]}]
print(parameters1)
RR = Ridge()
Grid1 = GridSearchCV(RR, parameters1,cv=4)
Grid1.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_data)
BestRR=Grid1.best_estimator_
print('Best Alpha for Ridge Regression: ', BestRR)
print('R^2 for the best Alpha: ', BestRR.score(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_test))
