#encoding=utf-8
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import naive_bayes
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import svm
from collections import Counter
import matplotlib.pyplot as plt
import lightgbm as lgb
import time
from multiprocessing import cpu_count

def modelLGB(trainX, trainY, testX, testY):

    # max_depth 最佳为9
    clf = lgb.LGBMClassifier(
                            boosting_type='gbdt', num_leaves=50, reg_alpha=0.0, reg_lambda=1.0,
                            max_depth=9, n_estimators=500, objective='binary',
                            subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
                            learning_rate=0.05, min_child_weight=10,random_state=2018,n_jobs=cpu_count()-1, min_split_gain=0.0000001
                            )
    clf.fit(trainX, trainY)

    predsTest = clf.predict(testX)
    predsTrain = clf.predict(trainX)

    accTrain = 0.0
    accTest = 0.0
    for idx in range(len(predsTrain)):
        if int(predsTrain[idx]) == trainY[idx]:
            accTrain += 1
    for idx in range(len(predsTest)):
        if int(predsTest[idx]) == testY[idx]:
            accTest += 1
    print ('lightgbm train accuracy: ', accTrain / len(trainX))
    print ('lightgbm test accuracy: ', accTest / len(testX))
    print clf.feature_importances_
    print clf.n_features_
    print featureKeys
    # plt.xlabel('Features')
    # plt.ylabel('Importance Score')
    # plt.bar(featureKeys, clf.feature_importances_ )
    # plt.show()

def modelLogit(trainX, trainY, testX, testY):
    clf = LogisticRegression(penalty='l2', C=1)
    clf.fit(trainX, trainY)

    predsTest = clf.predict(testX)
    predsTrain = clf.predict(trainX)

    accTrain = 0.0
    accTest = 0.0
    for idx in range(len(predsTrain)):
        if int(predsTrain[idx]) == trainY[idx]:
            accTrain += 1
    for idx in range(len(predsTest)):
        if int(predsTest[idx]) == testY[idx]:
            accTest += 1
    print ('svm train accuracy: ', accTrain / len(trainX))
    print ('svm test accuracy: ', accTest / len(testX))

def modelSVM(trainX, trainY, testX, testY):
    clf = svm.SVC(kernel='rbf', C=1.6)
    clf.fit(trainX, trainY)

    predsTest = clf.predict(testX)
    predsTrain = clf.predict(trainX)

    accTrain = 0.0
    accTest = 0.0
    for idx in range(len(predsTrain)):
        if int(predsTrain[idx]) == trainY[idx]:
            accTrain += 1
    for idx in range(len(predsTest)):
        if int(predsTest[idx]) == testY[idx]:
            accTest += 1
    print ('svm train accuracy: ', accTrain / len(trainX))
    print ('svm test accuracy: ', accTest / len(testX))

t = time.time()
trainData = pd.read_excel('shuju.xlsx', sheet_name='train')
# 将用途数值化，形成19维01变量
wayArray = np.zeros((len(trainData[u'用途']), 19))
wayType = Counter(trainData[u'用途']).keys()   #有多少种用途
wayDict = {}
k = 0
for key in wayType:  #将用途名称用数字变量替代
    wayDict[key] = k
    k += 1
wayNum = []
for idx in range(len(trainData[u'用途'])):
    wayNum.append(wayDict[trainData[u'用途'][idx]])
for idx in range(len(wayNum)):
    wayArray[idx][wayNum[idx]] = 1

name = u'way'
# for idx in range(wayArray.shape[1]):
#     trainData[name+str(idx)] = wayArray[:, idx]

print trainData.keys()

# trainData['wayNum'] = np.array(wayNum)

testData = pd.read_excel('shuju.xlsx', sheet_name='test')

'''
age 和 interest 一起放进去提升很大
'''

#去掉年龄 用途等特征 以下是经过选择的最优特征

removeFeature = [u'project', u'user', u'ID', u'recovery', u'default1', u'time', u'用途', u'thank', u'Amount',
                 u'symbol',u'lenth', u'repeat', u'positive', u'fog', u'symbol2', u'purpose', u'Marry']    #0.8805

featureKeys = []

keys = [u'Amount', u'Interest', u'Month', u'Credit', u'lenth', u'repeat', u'symbol',
        u'symbol2', u'fog', u'positive', u'work', u'earning', u'explain', u'assure',
        u'thank', u'please', u'purpose1', u'purpose2', u'purpose3', u'purpose4', u'purpose5',
        u'purpose6', u'purpose', u'Age', u'Edu', u'Marry', u'Income', u'Worktime', u'car', u'house', u'cloan', u'hloan']

for key in trainData.keys():
    if key not in removeFeature:
        featureKeys.append(key)
print featureKeys  #剩下的特征变量

X = trainData[featureKeys]

#归一化 采用zero均值标准化
eps = 0.0000001
#X = (X - X.min())/(X.max() - X.min() + eps)
#X = (X-X.mean())/X.std()
Y = np.array(trainData['default1'])

trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.1, random_state=42)

# print trainX.head()
print 'train begin -------- '

modelLGB(trainX, trainY, testX, testY)

print 'time: ', time.time() - t
