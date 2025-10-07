# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the libraries and read the data frame using pandas.

2.Calculate the null values present in the dataset and apply label encoder. 

3.Determine test and training data set and apply decison tree regression in dataset.

4.Calculate Mean square error,data prediction and r2.

## Program:
```python
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: KESHAVARTHINI B
RegisterNumber:212224040158


import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])

```

## Output:

<img width="432" height="268" alt="image" src="https://github.com/user-attachments/assets/8cd9f9fa-2ab8-4583-9f50-45bfdc8520e4" />.

<img width="507" height="236" alt="image" src="https://github.com/user-attachments/assets/d9004663-ecf7-4a55-ab86-794c2ba67a84" />.

<img width="245" height="97" alt="image" src="https://github.com/user-attachments/assets/2435669d-750b-405a-ba03-b10847dc9ce1" />.

<img width="397" height="252" alt="image" src="https://github.com/user-attachments/assets/c8c113ed-4ecb-4c9c-957c-6426bcc63821" />.

<img width="590" height="260" alt="image" src="https://github.com/user-attachments/assets/62d8e52e-3756-4882-bbc5-d9ec7de9f4d6" />.

<img width="317" height="67" alt="image" src="https://github.com/user-attachments/assets/3579d849-73ab-46ff-97c9-9d3ad9bbae76" />.

<img width="222" height="118" alt="image" src="https://github.com/user-attachments/assets/a5541bb0-97e0-4dea-8834-a45c03febf45" />.

<img width="1011" height="485" alt="image" src="https://github.com/user-attachments/assets/334904eb-3393-4705-974f-b310f57b6523" />.





## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
