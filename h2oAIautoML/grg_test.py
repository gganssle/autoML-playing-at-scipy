import h2o
from h2o.automl import H2OAutoML

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

h2o.init()

digits = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target)

x = ['C1', 'C2', 'C3', 'C4']
y = "C5"

# For binary classification, response should be a factor
y_train = y_train.asfactor()
y_test = y_test.asfactor()

X_train.shape
np.expand_dims(y_train, axis=1).shape
train = np.concatenate((X_train, np.expand_dims(y_train, axis=1)), axis=1)

train = h2o.H2OFrame(train)

# Run AutoML for 30 seconds
aml = H2OAutoML(max_runtime_secs = 30)
aml.train(x = x, y = y, training_frame = train)

# View the AutoML Leaderboard
lb = aml.leaderboard
lb

# The leader model is stored here
aml.leader

preds = aml.predict(test)

# or:
preds = aml.leader.predict(test)
preds
