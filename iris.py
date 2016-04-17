import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()

#print iris.feature_names
#print iris.target_names
#print iris.data[0]
#print iris.target[0]

# print out the whole dataset
#for i in range(len(iris.target)):
#    print "Example %d: label %s, feature %s" % (i, iris.target_names[iris.target[i]], iris.data[i])

test_idx = [0, 50, 100]

# training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data,  train_target)

print test_target
print clf.predict(test_data[2])
