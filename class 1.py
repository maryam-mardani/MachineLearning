import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston

from statsmodels.api import OLS, add_constant

boston = load_boston()
x = boston['data']
y = boston['target']

x_copy = pd.DataFrame(x, columns=boston['feature_names'])
y_copy = pd.Series(y, name="MEDV")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

print(x_train)
