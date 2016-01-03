import pandas
import json
import xgboost
import math
from score import quadratic_weighted_kappa
from scipy.optimize import minimize
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, Imputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation

# takes pandas DataFrame and returns sklearnable matrix
def preprocess(data):
    print "Preprocessing"
    categoricals = list()
    data = pandas.get_dummies(data)
    #for feat in ordinals:
        #data[feat + 'sq'] = data[feat] * data[feat]
    cols = data.columns
    for i, col in enumerate(cols):
        datatype = str(type(data[col][0]))
        unique_vals = len(data[col].unique())
        if unique_vals < 12 and 'float' not in datatype and False:
            categoricals.append(i)
    data = Imputer(strategy='most_frequent').fit_transform(data)
    mat = OneHotEncoder(categorical_features=categoricals).fit_transform(data) 
    print "Preprocessing complete"
    return mat

# returns predictor
def train(data, labels, params={}):
    print "Training"
    clf = LinearRegression(**params)
    clf.fit(data, labels)
    print "Training complete"
    return clf

def rmse(preds, actual):
    rmse = 0
    for i in range(0, len(preds)):
        se = (preds[i] - actual[i] / float(actual[i])) ** 2
        rmse += se
    rmse /= float(len(preds))
    return math.sqrt(rmse)

# returns prediction accuracy 
def validate(data, labels, predictor): 
    preds = predictor.predict(data)
    score = rmse(preds, labels)
    print "Accuracy with {}: {}".format(type(predictor).__name__, score)
    return score

def normalize_preds(preds, cutoffs):
    np = [8] * len(preds)
    for i, p in enumerate(preds):
        for j, c in enumerate(cutoffs):
            if p < c:
                np[i] = int(j + 1)
                break
    return np

# writes predictions for data to filename
def output(data, ids, predictor, filename, cutoffs):
    preds = normalize_preds(predictor.predict(data), cutoffs)
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

    bst = None
    for i in range(0, 1):
        total_data = preprocess(pandas.concat([train_df, test_df], ignore_index=True))#.todense()
        train_mat = total_data[0:len(labels)]
        test_mat = total_data[len(labels):]

        #itrain_X, val_X, itrain_Y, val_Y = cross_validation.train_test_split(train_mat, labels, 
            #test_size=0.25, random_state=5)

        #xgtrain = xgboost.DMatrix(itrain_X, label=itrain_Y)
        #xgval = xgboost.DMatrix(val_X, label=val_Y)
        #watchlist = [(xgtrain, 'train'), (xgval, 'val')]
        xgtrain = xgboost.DMatrix(train_mat, label=labels)
        xgtest = xgboost.DMatrix(test_mat)
        watchlist = [(xgtrain, 'train')]
        params = dict(objective="reg:linear", eta=0.03, max_depth=9, silent=1,
                      subsample=0.9, colsample_bytree=1)
        print "Training"
        bst = xgboost.train(params, xgtrain, 500, watchlist)

        #def to_minimize(cutoffs, preds, labels):
            #return -1 * quadratic_weighted_kappa(normalize_preds(preds, cutoffs), labels)
       
        #preds = bst.predict(xgval)
        #cutoffs = [1.64633795596, 2.96114099488, 3.44907543537, 4.32684581774, 5.69615093088, 6.693580911, 6.9198950906]
        #cutoffs = [1.79693398986, 3.25936723323, 4.17903236969, 4.76921656982, 5.17561713721, 5.94390027511, 6.39053004388]
        #cutoffs = [1.80938857862, 3.39480207164, 4.17896052705, 4.76215476766, 5.25429867111, 5.95061434557, 6.387262249705]
        #cutoffs = [1.87805701095, 3.43585154288, 4.17881664995, 4.76230850377, 5.25319919262, 5.95010648948, 6.38726431059]
        cutoffs = [1.98616053055, 3.55054541238, 3.91550873252, 4.82133899229, 5.69285710923, 6.16656574304, 6.78020937551]

        #cutoffs = [1.6, 2.8, 3.2, 4.2, 5.3, 6.7, 7.3]

        #obj = minimize(to_minimize, cutoffs, args=(preds, val_Y), method='Nelder-Mead')
        #cutoffs = obj.x
        #print "{} {} New cutoffs: {}".format(obj.success, obj.message, ' '.join([str(c) for c in cutoffs]))

        #preds = normalize_preds(bst.predict(xgval), cutoffs)
        #print quadratic_weighted_kappa(preds, val_Y)

        #train_df['xgOut{}'.format(i)] = [round(a) for a in bst.predict(xgboost.DMatrix(train_mat))]
        #test_df['xgOut{}'.format(i)] = [round(a) for a in bst.predict(xgboost.DMatrix(test_mat))]
    
        output(xgtest, test_ids, bst, 'boostedXG-md9-cut-{}.csv'.format(i), cutoffs)


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

ordinals = ['Product_Info_4', 'Ins_Age', 'Ht', 'Wt', 'BMI', 'Employment_Info_1', 'Employment_Info_4',
    'Employment_Info_6', 'Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3',
    'Family_Hist_4', 'Family_Hist_5', 'Medical_History_1', 'Medical_History_10', 
    'Medical_History_15', 'Medical_History_24', 'Medical_History_32']
