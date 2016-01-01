import pandas
import json
import xgboost
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, Imputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn import cross_validation

lf = ['Wt', 'BMI', 'Ins_Age', 'Medical_History_15', 'Product_Info_4']
cat_feats = ['Medical_History_39', 'Medical_History_5', 'Medical_History_18',
             'Medical_History_13', 'InsuredInfo_7', 'Medical_History_9', 
             'Medical_History_38', 'InsuredInfo_5', 'Medical_History_22', 
            'InsuredInfo_2', 'Medical_History_7', 'Medical_History_14', 
            'Medical_History_27', 'Medical_History_33', 'Medical_History_30', 
            'Medical_History_35', 'Employment_Info_3', 'Medical_History_8', 
            'Medical_History_4', 'Family_Hist_1', 'Insurance_History_9', 
            'Medical_History_21', 'Medical_History_19', 'InsuredInfo_4', 
            'Medical_History_36', 'Medical_History_11', 'Product_Info_2', 
            'Medical_History_25', 'InsuredInfo_3', 'Insurance_History_8', 
            'Insurance_History_1', 'InsuredInfo_6', 'Insurance_History_4',
            'Medical_History_3', 'Product_Info_6', 'Insurance_History_7',
            'Medical_History_20', 'Medical_History_6', 'Medical_History_31', 
            'Employment_Info_5', 'Insurance_History_2', 'Product_Info_1', 
            'Product_Info_5', 'Medical_History_17', 'InsuredInfo_1', 
            'Employment_Info_2', 'Medical_History_28', 'Medical_History_23',
            'Medical_History_12', 'Medical_History_16', 'Medical_History_40', 
            'Product_Info_3', 'Medical_History_37', 'Medical_History_29', 
            'Medical_History_34', 'Medical_History_41', 'Product_Info_7', 
            'Medical_History_26', 'Insurance_History_3', 'Medical_History_2']

# takes pandas DataFrame and returns sklearnable matrix
def preprocess(data):
    print "Preprocessing"
    categoricals = list()
    #data = data[lf + cat_feats[-5:]]
    data = pandas.get_dummies(data)
    #data = data.fillna(0)
    cols = data.columns
    for i, col in enumerate(cols):
        datatype = str(type(data[col][0]))
        unique_vals = len(data[col].unique())
        if unique_vals < 12 and 'float' not in datatype:
            categoricals.append(i)
    data = Imputer(strategy='most_frequent').fit_transform(data)
    mat = OneHotEncoder(categorical_features=categoricals).fit_transform(data) 
    print "Preprocessing complete"
    return mat

# returns predictor
def train(data, labels, params={}):
    print "Training"
    clf = SVC(**params)
    clf.fit(data, labels)
    print "Training complete"
    return clf

# returns prediction accuracy 
def validate(data, labels, predictor): 
    score = predictor.score(data, labels)
    print "Accuracy with {}: {}".format(type(predictor).__name__, score)
    return score
            

# writes predictions for data to filename
def output(data, ids, predictor, filename, regression=False):
    preds = predictor.predict(data)
    if regression:
        for i, p in enumerate(preds):
            p = round(p)
            if p < 1: p = 1
            elif p > 8: p = 8
            preds[i] = int(p)
    with open(filename, "w") as f:
        f.write('"Id","Response"\n')
        for i, p in enumerate(preds):
            f.write("{},{}\n".format(ids[i], int(p)))

 
if __name__ == "__main__": 
    print "Reading CSVs"
    train_df = pandas.read_csv('train.csv')
    test_df = pandas.read_csv('test.csv')
    print "CSV reading complete"

    labels = train_df['Response'].get_values()
    columns = list(train_df.columns)
    columns.remove('Response')
    columns.remove('Id')
    train_df = train_df[columns] # remove Response, Id variables from training data
    test_ids = test_df['Id'].get_values() # get a list of Id values for use in creating submission
    test_df = test_df[columns] # remove Id variable from test data

    total_data = preprocess(pandas.concat([train_df, test_df], ignore_index=True)).todense()
    train_mat = total_data[0:len(labels)]
    test_mat = total_data[len(labels):]

    itrain_X, val_X, itrain_Y, val_Y = cross_validation.train_test_split(train_mat, labels, test_size=0.25)
    xgtrain = xgboost.DMatrix(itrain_X, label=itrain_Y)
    xgval = xgboost.DMatrix(val_X, label=val_Y)
    watchlist = [(xgtrain, 'train'), (xgval, 'val')]
    #xgtrain = xgboost.DMatrix(train_mat, label=labels)
    #xgtest = xgboost.DMatrix(test_mat)
    #watchlist = [(xgtrain, 'train')]
    params = dict(objective="reg:linear", eta=0.03, max_depth=7, silent=1,
                  subsample=0.9, colsample_bytree=1)
    print "Training"
    bst = xgboost.train(params, xgtrain, 680, watchlist)

    output(xgtest, test_ids, bst, 'xgregOutput.csv', regression=True)
    #pred = bst.predict(xgtest)

    """
    predictor = train(train_mat, labels, dict(n_estimators=300))
    output(test_mat, test_ids, predictor, 'RFoutput.csv')
    """
