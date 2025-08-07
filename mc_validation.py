"""
This is a short script that I wrote to test a models preformance on different train test split scenarios,
which is nescesary due to the small dataset size.

parameters:
    mymodel: ML model that will be tested
          X: Predictive Variables
          y: target variable
   num_sims: Number of times
"""
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
def mc_validation(mymodel,X,y,num_sims): #enter the model,predictive variables, target variable, and number of simulations

    score_list=[]
    for i in range (1,num_sims): #iterating through different random_states for num_sims ammount of times
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=i) 
        mymodel.fit(X_train,y_train) #fitting model
        
        prediction = mymodel.predict(X_test) #making predictions
        score = accuracy_score(y_test,prediction) #producing accuracy score for random_state instance i
        
        score_list.append(score) #appending the results to a list
        
    score_mean = np.mean(score_list) #getting the average
    score_std = np.std(score_list) 
    print(f"mean accuracy: {score_mean}")
    print(f"standard deviation of scores: {score_std}") 
    return score_mean, score_std
