from sklearn.tree import DecisionTreeClassifier
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
iris=load_iris()
x=iris.data[:,[2,3]]
y=iris.target
model=DecisionTreeClassifier(criterion='entropy',max_depth=3,random_state=1)
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=1)
model.fit(xtrain,ytrain)
model.predict(xtest)
plot_decision_regions(x,y,clf=model)
plt.show()