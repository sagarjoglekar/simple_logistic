
# A super basic Logistic regressor built just using numpy


## logistic/logistic.py 
this file contains the entire implementation for the logistic regression along with the SGD based optimization. 
The LogisticRegression class encapsulates everything. 

## main_driver.py

If you do not want to go through a notebook and just test the logistic regression, you can just run the this file and it will do a 5 fold cross validation
with a L2 regularization. You should get an AUC of around 0.75 and an F1-score of around 0.75

## dependencies.txt 
List of dependencies if you want to run it in a virtual env. But I think all you need are pandas , numpy, and sklearn. All the others are pulled in due to 
jupyter.