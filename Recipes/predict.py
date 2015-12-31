import json
import xgboost
from sklearn import linear_model
from sklearn import feature_extraction
from sklearn.naive_bayes import GaussianNB
from sklearn import ensemble
from sklearn import discriminant_analysis
from sklearn import neighbors
from sklearn import decomposition

with open('train.json') as f:
    data = json.load(f)

with open('test.json') as f:
    test = json.load(f)

print "files read in"

def createMatrices(data_json):
    docs = [' '.join([x.replace(" ", " ") for x in d['ingredients']]) for d in data_json]
    extractor = feature_extraction.text.TfidfVectorizer(ngram_range=(1,1))
    mat = extractor.fit_transform(docs)
    return mat

all_X = createMatrices(data + test)
data_X = all_X[0:len(data)]
test_X = all_X[len(data):]
data_Y = [d['cuisine'] for d in data]

data_X = decomposition.PCA(n_components=100).fit_transform(data_X.toarray())
ss = data_X.shape[0] * 80 / 100
train_X = data_X[0:ss]
train_Y = data_Y[0:ss]
val_X = data_X[ss:]
val_Y = data_Y[ss:]

lr = linear_model.LogisticRegression(C=5)
lr.fit(train_X, train_Y)

threshold = 8
correct = 0
for i, row in enumerate(val_X):
    pred = lr.predict(row)
    if pred == val_Y[i]:
        correct += 1
print "Accuracy: {}".format(correct / float(len(val_Y)))



"""
print "Linear Discriminant Analysis"
lda = discriminant_analysis.LinearDiscriminantAnalysis()
lda.fit(train_X.toarray(), train_Y)
print lda.score(val_X.toarray(), val_Y)

lr = linear_model.LogisticRegression(C=5)
lr.fit(data_X, data_Y)
print "done training"

classes = set()
for y in data_Y: classes.add(y)
class_count = len(classes)
class_map = {}
counter = 0
for c in classes:
    class_map[c] = float(counter)
    counter += 1

print "XGBoost"

xgtrain_Y = [class_map[y] for y in train_Y]
xgtest_Y = [class_map[y] for y in val_Y]
xgtrain = xgboost.DMatrix(train_X, label=xgtrain_Y)
xgtest = xgboost.DMatrix(val_X, label=xgtest_Y)
watchlist = [(xgtrain, 'train'), (xgtest, 'test')]
params = dict(objective="multi:softmax", eta=0.1, max_depth=6, silent=1, num_class=class_count,
              subsample=1, colsample_bytree=1)
print "Training"
bst = xgboost.train(params, xgtrain, 1000, watchlist)
pred = bst.predict(xgtest)
print "Accuracy = {}".format(sum(int(pred[i] == xgtest_Y[i]) for i in range(0, len(pred))) / float(len(pred)))

Cs = [5]
print "Logistic Regression"
for c in Cs:
    print "C = {}".format(c)
    lr = linear_model.LogisticRegression(C=c)
    lr.fit(train_X, train_Y)
    print lr.score(val_X, val_Y)
"""
