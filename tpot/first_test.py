from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

clf = LogisticRegression()
clf.fit(X_train, y_train)
clf.predict(X_test)
