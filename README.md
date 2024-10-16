# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas and read the Employee dataset
2. Check for any null values and preprocess the data
3. Split the data into Training dataset and Testing dataset
4. Import DecisionTreeClassifier from sklearn.tree and train the model
5. Test the model using the Testing dataset
6. Find the accuracy of the model through importing metrics from sklearn (accuracy_score)
7. Give input to the model and predict the results

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Sujithra D
RegisterNumber: 212222220052
*/
import pandas as pd
data=pd.read_csv('/content/Employee (1).csv')
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
y.head()


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)


from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy


dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:

![image](https://github.com/user-attachments/assets/d9bd2d01-066b-4175-aece-9d53c696366d)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
