##!
import sys
sys.path.append('/home/alvaro/Alvaro/Storage/AlvResearch/Software/LiClipse/AlvaroWorkspace/Test')   
import AsaUtils

import initOptions
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

#Section 0. Define Path to seek features
RootDescPath='MIT-67-DescriptorsPath'

#Section 1. Define classifiers to use in this activity
classifiersDefs = {'KNN':"KNeighborsClassifier(n_neighbors=5, metric='euclidean', weights='uniform')",
                   'SVM':"svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr')",
                   'NNS':"MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100, 100), learning_rate='constant', learning_rate_init=0.001, shuffle=True, random_state=1)"}

#Section 2. Init libraries and main options
options=initOptions.selections()
 
#Section 3. Read training data
descriptorsPath=options[RootDescPath] + 'TrainSet'
classNames,trainFeats,trainLabels=AsaUtils.getMatlabDescriptors(descriptorsPath, 'mat',4096)

#Section 4. Train Classifier
print('******Training classifier, it can take time !, be patient !')
classifier = eval(classifiersDefs['SVM'])

classifier.fit(trainFeats,trainLabels)
    
#Section 5. Read test data
descriptorsPath=options[RootDescPath] + 'TestSet'
classNames,testFeats,testLabels=AsaUtils.getMatlabDescriptors(descriptorsPath, 'mat',4096)

#Section 6. Apply Classifier to test data and calculate accuracy 
print('******Testing classifier, it can take time !, be patient !')
predictedLabels=classifier.predict(testFeats)
accuracy=accuracy_score(testLabels, predictedLabels)
print 'Classification accuracy on test set: ' + str(100*round(accuracy,2)) + '%'

#Section 7. Apply Classifier to train data and calculate accuracy 
predictedTrainLabels=classifier.predict(trainFeats)
accuracy=accuracy_score(trainLabels, predictedTrainLabels)
print 'Classification accuracy on training set: ' + str(100*round(accuracy,2)) + '%'



    
    



    
    
