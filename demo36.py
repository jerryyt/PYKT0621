from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np

URL1 = "http://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
df = pd.read_csv(URL1, header=None, prefix="X")
print(df.shape)

# cut data
data, labels = df.iloc[:, :-1], df.iloc[:, -1]
print(data.shape)
print(labels.shape)
df.rename(columns={'X60': 'Lable'}, inplace=True)
print(df.shape)

classifier = KNeighborsClassifier(n_neighbors=4)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)


classifier.fit(x_train, y_train)
y_predict = classifier.predict(x_test)
print(f"result={classifier.score(x_test, y_test)}")
print(y_predict)
print(y_test)
result = confusion_matrix(y_test, y_predict)
print(result)

scores = cross_val_score(classifier, data, labels, cv=5, groups=labels)
print(scores, np.mean(scores))