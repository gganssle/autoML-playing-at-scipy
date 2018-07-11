from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from tpot import TPOTClassifier

digits = datasets.load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

clf = LogisticRegression()
clf.fit(X_train, y_train)
accuracy_score(y_test, clf.predict(X_test))

tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2, n_jobs=-1)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
