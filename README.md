# Credit Card Default

Credit card default happens when you have become severely delinquent on your credit card payments. Default is a serious credit card status that affects not only your standing with that credit card issuer but also your credit standing in general and your ability to get approved for other credit-based services.


## Installation and importing libraries of python package



```bash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

## Read the data 

```
data=pd.read_csv('E:/dat/creditcard.csv')

### reading first 5 rows data ###
data.head()

### Information of data ###
data.info()

### Checking any NaN values in dataset ###
data.isnull().sum()

```


## Imbalanced Data

```
data['Class'].value_counts()

## Return a random sample of items from an axis of object ###
legit_sample=legit.sample(n=492)
legit_sample.head()
```

## Spliting the dataset
```
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=1)
```
## Model Building
```
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
```
## Model Evaluation
```
from sklearn import metrics
```
## Hyper Parameter
```
Logistic regression does have not really have any critical hyperparameters to tune.

Sometime few of them are imp. which can inhace the performance

->solver in [‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’]

->penalty in [‘none’, ‘l1’, ‘l2’, ‘elasticnet’]

->C in [100, 10, 1.0, 0.1, 0.01] C parameter controls the penality strength, which can also be effective

for all parameter of Logistic Regession :https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
```
## Model Saving
```
import pickle

with open("model.pkl", "wb") as f:
    pickle.dump(grid_model, f)

```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
