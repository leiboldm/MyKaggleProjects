import pandas
import json
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, Imputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation

# takes pandas DataFrame and returns sklearnable matrix
def preprocess(data):
    print "Preprocessing"
    categoricals = list()
    data = pandas.get_dummies(data)
    cols = data.columns
    for i, col in enumerate(cols):
        datatype = str(type(data[col][0]))
        unique_vals = len(data[col].unique())
        if unique_vals < 12 and 'float' not in datatype:
            categoricals.append(i)
    data = Imputer().fit_transform(data)
    mat = OneHotEncoder(categorical_features=categoricals).fit_transform(data) 
    print "Preprocessing complete"
    return mat

# returns predictor
def train(data, labels, params={}):
    print "Training"
    clf = RandomForestClassifier(**params)
    clf.fit(data, labels)
    print "Training complete"
    return clf

# returns prediction accuracy 
def validate(data, labels, predictor): 
    score = predictor.score(data, labels)
    print "Accuracy with {}: {}".format(type(predictor).__name__, score)
    return score
            

# writes predictions for data to filename
def output(data, ids, predictor, filename):
    preds = predictor.predict(data)
    with open(filename, "w") as f:
        f.write('"Id","Response"\n')
        for i, p in enumerate(preds):
            f.write("{},{}\n".format(ids[i], p))
 
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
    Ns = [3, 5, 10, 25, 100, 300, 1000]
    for n in Ns:
        print n
        params = dict(n_estimators=n, verbose=1)
        predictor = train(itrain_X, itrain_Y, params)
        validate(val_X, val_Y, predictor)

    """
    predictor = train(train_mat, labels, dict(n_estimators=300))
    output(test_mat, test_ids, predictor, 'RFoutput.csv')
    """
