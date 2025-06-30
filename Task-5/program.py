import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
import os
os.environ["PATH"] += os.pathsep + r'D:\Program Files\Graphviz\bin'
data = pd.read_csv('heart.csv')
print(data.head())
print(data.info())
print(data.isnull().sum())
print(data['target'].value_counts())
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
print("Training accuracy:", model.score(X_train, y_train))
print("Test accuracy:", model.score(X_test, y_test))

dot_data = export_graphviz(model, out_file=None, feature_names=X.columns,
                           class_names=['No Disease', 'Disease'], filled=True)
graph = graphviz.Source(dot_data)
graph.render("decision_tree")
graph.view()
model_limited = DecisionTreeClassifier(max_depth=4, random_state=42)
model_limited.fit(X_train, y_train)
print("Test accuracy (limited depth):", model_limited.score(X_test, y_test))

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

print("Random Forest accuracy:", rf.score(X_test, y_test))

importances = pd.Series(rf.feature_importances_, index=X.columns)
importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Important Features")
plt.show()

scores = cross_val_score(rf, X, y, cv=5)
print("Cross-validated accuracy:", scores.mean())
