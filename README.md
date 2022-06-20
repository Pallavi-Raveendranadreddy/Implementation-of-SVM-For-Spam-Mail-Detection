# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
### Step1:Import the required packages.
### Step2:Import the dataset to operate on.
### Step3:Split the dataset.
### Step4:Predict the required output.
### Step5:End the program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Valasareddy Pallavi
RegisterNumber:212221240059  
*/
```
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/content/Spam.csv',encoding='latin-1')
df = data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)

df.head()

df.info()

df.isnull().sum()

x=df["v1"].values
y=df["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test = cv.transform(x_test)

from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train,y_train)

y_pred = svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:
<img width="535" alt="c8" src="https://user-images.githubusercontent.com/94294872/174632568-fdfbf874-de8b-4c2c-a9ea-b78fdc31226c.png">

<img width="349" alt="c9" src="https://user-images.githubusercontent.com/94294872/174632653-2bf7de18-e39b-4822-ba60-04450052e770.png">

<img width="272" alt="c10" src="https://user-images.githubusercontent.com/94294872/174632696-8f6e88bf-b608-407f-be13-e8b1029a4390.png">

<img width="430" alt="c11" src="https://user-images.githubusercontent.com/94294872/174632736-5a20b040-6830-4175-a00c-62ca23705120.png">

<img width="248" alt="c12" src="https://user-images.githubusercontent.com/94294872/174632765-e0b3ad90-d560-4b70-81e2-0ae04719311b.png">


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
