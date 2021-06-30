import numpy as np
import pandas as pd    
import tensorflow as tf
from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

cifar10= tf.keras.datasets.cifar10
(x_train,y_train), (x_test, y_test) = cifar10.load_data()
x_train_2d = x_train.reshape(50000,3072)
x_test_2d = x_test.reshape(10000, 3072)

pca = PCA()
pca.fit(x_train_2d)
k = 0  
total = sum(pca.explained_variance_)
current_sum = 0
while current_sum/total < 0.99:
    current_sum += pca.explained_variance_[k]
    k = k + 1

pca=PCA(n_components=k,whiten=True)
train_transform_data= pca.fit_transform(x_train_2d)
test_transform_data = pca.transform(x_test_2d)


x_train1,x_test1,y_train1,y_test1= train_test_split(train_transform_data,y_train) 

clf = svm.SVC()
grid = { 'C' : [1e2,1e3,5e3,1e4,1e5],
         'gamma' : [1e-3,5e-4,1e-4,5e-3]}
abc = GridSearchCV(clf,grid)
abc.fit(x_train1,y_train1)
abc.best_estimator_

clf_svm = svm.SVC()
clf_svm.fit(train_transform_data, y_train)
clf.score(test_transform_data,y_test1)
y_pred=clf.predict(y_test)

final_prediction= pd.DataFrame(y_pred)
final_prediction.to_csv('Final_pred.csv', index = False, header = False)



