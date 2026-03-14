# Implementation of Multivariate Linear Regression
## Aim
To write a python program to implement multivariate linear regression and predict the output.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner
## Algorithm:
### Step1
import pandas as pd.

### Step2
Read the csv file.

### Step3
Get the value of X and y variables

### Step4
Create the linear regression model and fit.

### Step5
Predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm cube

## Program:
```
~~~ python
name:suvetha.k
ref: 212225040444

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model,metrics
from sklearn.datasets import fetch_california_housing

california=fetch_california_housing()
x=california.data
y=california.target
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=1)
reg=linear_model.LinearRegression()
reg.fit(x_train,y_train)
print("COEFFICIENT:",reg.coef_)
print("\n")
print("VARIANCE SCORE: {}".format(reg.score(x_test,y_test)))
print("\n")
plt.style.use("fivethirtyeight")

plt.scatter(reg.predict(x_train),
            reg.predict(x_train) - y_train,
            color="",
            s=10,
            label="TRAIN_DATA")

plt.scatter(reg.predict(x_test),
            reg.predict(x_test) - y_test,
            color="green",
            s=10,
            label="TEST_DATA")

plt.axhline(y=0, linewidth=2)
plt.legend(loc='upper right')
plt.title('Residual Errors')
plt.show()
~~~
## Output:

<img width="962" height="722" alt="Screenshot 2026-02-27 104133" src="https://github.com/user-attachments/assets/1eaa2bf3-187f-42ae-8305-d7fd27927f38" />

### Insert your output


<img width="895" height="764" alt="Screenshot 2026-02-27 104148" src="https://github.com/user-attachments/assets/eacc02e2-010f-44ec-bc8c-c3430605d649" />


## Result
Thus the multivariate linear regression is implemented and predicted the output using python program.







