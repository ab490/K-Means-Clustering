#Perform classification using k-means clustering for the given data
#Linearly Separable Data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler,MinMaxScaler


path = "E:/Sem 5/BE502 Design and Analysis of Bioalgorithms/Assignment 3/ls_data"


class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self,data):

        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in data:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    optimized = False

            if optimized:
                break

    def predict(self,data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


def accuracy(y_true,y_pred):
    assert y_true.shape==y_pred.shape , "Shape Mismatch"
    correct=0
    for i in range(len(y_true)):
        if(y_true[i]==y_pred[i]):
            correct+=1
    return correct/len(y_true)


#Reading the data
ls_cls1=pd.read_csv(f"{path}/class1.txt",sep=',',names=["Col_1","Col_2"])
ls_cls1["class"]=[0 for i in range(len(ls_cls1))]

ls_cls2=pd.read_csv(f"{path}/class2.txt",sep=',',names=["Col_1","Col_2"])
ls_cls2["class"]=[1 for i in range(len(ls_cls2))]

ls_data=pd.concat([ls_cls1,ls_cls2])    
ls_data = ls_data.sample(frac=1,random_state=42).reset_index(drop=True)     


X_train,X_test,y_train,y_test=train_test_split(ls_data[["Col_1","Col_2"]],ls_data["class"],test_size=0.3,random_state=42)
X_train=np.array(X_train)
X_test=np.array(X_test)
y_train=np.array(y_train)
y_test=np.array(y_test)


kmeans=K_Means()
kmeans.fit(X_train)

centers=kmeans.centroids

center_0=centers[0]
center_1=centers[1]

center=np.vstack((center_0,center_1))         


y_train_pred=[]
for i in X_train:
    y_train_pred.append(kmeans.predict(i))
y_train_pred=np.array(y_train_pred)


y_test_pred=[]
for i in X_test:
    y_test_pred.append(kmeans.predict(i))
y_test_pred=np.array(y_test_pred)



#Results
print("Evaluating performance of the model:")
print("Training Accuracy on the LS dataset:", accuracy(y_train,y_train_pred))
print("Test Accuracy on the LS dataset:", accuracy(y_test,y_test_pred),"\n")

plt.scatter(center[:,0],center[:,1],color='red',label="centroids")
plt.scatter(ls_cls1["Col_1"],ls_cls1["Col_2"],label="class1")
plt.scatter(ls_cls2["Col_1"],ls_cls2["Col_2"],label="class2")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("Plot of Original Data")
plt.legend()
plt.show()

plt.scatter(center[:,0],center[:,1],color='red',label="centroids")
plt.scatter(X_train[:,0],X_train[:,1],c=y_train_pred)
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("Plot of Train Data Predicted")
plt.legend()
plt.show()

plt.scatter(center[:,0],center[:,1],color='red',label="centroids")
plt.scatter(X_test[:,0],X_test[:,1],c=y_test)
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("Plot of Original Test Data")
plt.legend()
plt.show()

plt.scatter(center[:,0],center[:,1],color='red',label="centroids")
plt.scatter(X_test[:,0],X_test[:,1],c=y_test_pred)
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("Plot of Test Data Predicted")
plt.legend()
plt.show()
