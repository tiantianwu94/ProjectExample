# -*- coding: utf-8 -*-

# Load libraries
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy as np
print('numpy: {}'.format(np.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))
import os
import pickle
import seaborn as sns
from os import chdir, getcwd
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import plotly
import plotly.plotly as py
import plotly.tools as tls
import time

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier



# Reset path
wd=getcwd()
chdir('C:\\Users\HP\\Desktop\\classes\\WT\\project1')  # need to change the path to run

# Load dataset
x1 = pd.ExcelFile('LendingClub_Data.xls')
data = x1.parse("Cleansed_Data_20180420")


# shape
print(data.shape)
# head
print(data.head(20))
# descriptions
print(data.describe())
# box and whisker plots
data.plot(kind='box', subplots=True, layout=(18,2), sharex=False, sharey=False)
plt.show()
# histograms
data.hist()
plt.show()


# count plot to see default rate
def add_freq():
    ncount = len(data)

    ax2=ax.twinx()

    ax2.yaxis.tick_left()
    ax.yaxis.tick_right()

    ax.yaxis.set_label_position('right')
    ax2.yaxis.set_label_position('left')

    ax2.set_ylabel('Frequency [%]')

    for p in ax.patches:
        x=p.get_bbox().get_points()[:,0]
        y=p.get_bbox().get_points()[1,1]
        ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 
                ha='center', va='bottom')

    ax2.set_ylim(0,100)
    ax2.grid(None)

ax = sns.countplot(x = data.Loan_Status_Final ,palette="Set3")
sns.set(font_scale=1.5)
ax.set_ylim(top = 50000)
ax.set_xlabel(' ')
ax.set_ylabel(' ')
fig = plt.gcf()
fig.set_size_inches(10,5)
ax.set_ylim(top=60000)

add_freq()
plt.show()


#### Cross Validatin 
dataset = data.fillna(lambda x: x.median())

Y = dataset.Loan_Status_Final.values
X = dataset.drop(['Loan_Status_Final', 'Purpose'], axis=1)
ScaledX = StandardScaler().fit_transform(X)

#array = dataset.values
#X = array[:,0:36]
#Y = array[:,36]

validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(ScaledX, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
models = []
#models.append(('LR', LogisticRegression()))
#models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('TREE', DecisionTreeClassifier()))
#models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('Boosting', AdaBoostClassifier()))
models.append(('ANN', MLPClassifier()))


# evaluate each model in turn
results = []
names = []
times = []
n_splits = 10
for name, model in models:
    kfold = model_selection.KFold(n_splits, random_state=seed)
    start_time = time.time()
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    run_time =  time.time() - start_time
    times.append(run_time)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

filehandler = open('cvResults.obj','wb')
pickle.dump(results, filehandler)
filehandler.close ()

filehandler = open('runtimeResults.obj','wb')
pickle.dump(times, filehandler)
filehandler.close ()

start_time = time.time()
print("--- %s seconds ---" % (time.time() - start_time))


# Compare accuracy by samplesize
infile = open('cvResults0.2.obj','rb')
cvResults02 = pickle.load(infile)
infile.close()

infile = open('cvResults0.3.obj','rb')
cvResults03 = pickle.load(infile)
infile.close()

infile = open('cvResults0.4.obj','rb')
cvResults04 = pickle.load(infile)
infile.close()

infile = open('cvResults0.5.obj','rb')
cvResults05 = pickle.load(infile)
infile.close()

infile = open('cvResults0.6.obj','rb')
cvResults06 = pickle.load(infile)
infile.close()

infile = open('cvResults0.7.obj','rb')
cvResults07 = pickle.load(infile)
infile.close()

infile = open('cvResults0.8.obj','rb')
cvResults08 = pickle.load(infile)
infile.close()

sampleSize = [8000,12000,16000,20000,24000,28000,32000]
aucMean =    [cvResults08[3].mean(),cvResults07[3].mean(),
              cvResults06[3].mean(),cvResults05[3].mean(),
              cvResults04[3].mean(),cvResults03[3].mean(),
              cvResults02[3].mean()]

fig = plt.figure()
plt.plot(sampleSize, aucMean, 'bo')  # plot x and y using blue circle markers
plt.plot(sampleSize, aucMean, linestyle='solid', label=names[3])  # plot x and y using blue circle markers
plt.title('sample size vs. model accuracy')
plt.xlabel('sampe size')
plt.ylabel('model accuracy')
plt.xlim(6000, 38000)
plt.ylim(0.82, 0.855)


aucMean1 =    [cvResults08[0].mean(),cvResults07[0].mean(),
              cvResults06[0].mean(),cvResults05[0].mean(),
              cvResults04[0].mean(),cvResults03[0].mean(),
              cvResults02[0].mean()]

plt.plot(sampleSize, aucMean1, 'yo')  # plot x and y using blue circle markers
plt.plot(sampleSize, aucMean1, linestyle='-', label=names[0])  # plot x and y using blue circle markers

aucMean1 =    [cvResults08[1].mean(),cvResults07[1].mean(),
              cvResults06[1].mean(),cvResults05[1].mean(),
              cvResults04[1].mean(),cvResults03[1].mean(),
              cvResults02[1].mean()]

plt.plot(sampleSize, aucMean1, 'ro')  # plot x and y using blue circle markers
plt.plot(sampleSize, aucMean1, linestyle=':', label=names[1])  # plot x and y using blue circle markers

aucMean1 =    [cvResults08[2].mean(),cvResults07[2].mean(),
              cvResults06[2].mean(),cvResults05[2].mean(),
              cvResults04[2].mean(),cvResults03[2].mean(),
              cvResults02[2].mean()]

plt.plot(sampleSize, aucMean1, 'go')  # plot x and y using blue circle markers
plt.plot(sampleSize, aucMean1, linestyle='-.', label=names[2])  # plot x and y using blue circle markers

aucMean1 =    [cvResults08[4].mean(),cvResults07[4].mean(),
              cvResults06[4].mean(),cvResults05[4].mean(),
              cvResults04[4].mean(),cvResults03[4].mean(),
              cvResults02[4].mean()]

plt.plot(sampleSize, aucMean1, 'ro')  # plot x and y using blue circle markers
plt.plot(sampleSize, aucMean1, linestyle='--', label=names[4])  # plot x and y using blue circle markers

plt.legend(loc="lower right")

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Method Comparison @ 80% population')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


fig = plt.figure()
fig.suptitle('Method Comparison @ 20% population')
ax = fig.add_subplot(111)
plt.boxplot(cvResults08)
ax.set_xticklabels(names)
plt.show()

# Compare Algorithms by running time

mpl_fig = plt.figure()
ax = mpl_fig.add_subplot(111)

nMethod = len(times)
xRange = range(nMethod)
width = 1/1.5
ax.bar(xRange, times, width, color="blue")
ax.set_title ('Compare Algorithms by running time')
ax.set_ylabel('seconds')
ax.set_xticklabels(('','KNN', 'TREE', 'SVM', 'Boosting', 'ANN'))


# Optimal Hyper Parameter Search
def svc_param_selection(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_

clf = svc_param_selection(X, y, nfolds)

           
# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

ann = MLPClassifier()
ann.fit(X_train, Y_train)
predictions = ann.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

svm = SVC()
svm.fit(X_train, Y_train)
predictions = svm.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

# ROC curve
def plotCvRocCurve(X, y, classifier, nfolds=5):
    
    from sklearn.metrics import roc_curve, auc
    from sklearn.cross_validation import StratifiedKFold
    import matplotlib.pyplot as plt
    from scipy import interp

    cv = StratifiedKFold(y, n_folds=nfolds)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []

    for i, (train, test) in enumerate(cv):
        probas_ = classifier.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])

        fpr, tpr, thresholds = roc_curve(y.iloc[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('CV ROC curve')
    plt.legend(loc="lower right")
    fig = plt.gcf()
    fig.set_size_inches(15,5)

    plt.show()



#plotCvRocCurve(X_train, Y_train, svm)
#plotCvRocCurve(X_train, Y_train, AdaBoostClassifier)

#y = dataset.Loan_Status_Final
#X = dataset.drop(['Loan_Status_Final', 'Purpose'], axis=1)
#X = StandardScaler().fit_transform(X)

#plotCvRocCurve(X, y, AdaBoostClassifier)


#X = X_train
#y = Y_train
#classifier = svm

data = data.sample(5000)
X = data.drop(['Loan_Status_Final', 'Purpose'], axis=1)
y = data.Loan_Status_Final

random_state = np.random.RandomState(0)



# try different boostin
classifier =  AdaBoostClassifier(DecisionTreeClassifier(max_depth=4, min_samples_split=20, min_samples_leaf=5),
                         algorithm="SAMME.R",
                         n_estimators=2000, learning_rate=0.5)

classifier = AdaBoostClassifier()

# try different SVC
classifier = svm.SVC(kernel='linear', probability=True,
                     random_state=random_state)
classifier = SVC()
classifier = svm.SVC(kernel='rbf', probability=True,
                     random_state=random_state)
nfolds=2
plotCvRocCurve(X, y, classifier, nfolds)