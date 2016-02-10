import numpy
import pandas
import xgboost
import re
import sys
import time
from sklearn import cross_validation
from sklearn.metrics import roc_auc_score

def preprocess(data): 
    return data    

print "Reading CSVs"
def convertDD(tf):
    if tf[0] == 'T': return 1
    else: return 0

def convertRAIL(rail):
    if rail == "000000": return 0
    else: return 1

highway_re = re.compile(r"I-\d+|hwy|highway|interstate")
def convertTWAYID(twayid):
    if re.search(highway_re, twayid):
        return 0
    if ('main' in twayid) or ('MAIN' in twayid):
        return 1
    else:
        return 2

columns_to_remove = ["DRUNK_DR", "ID", "LONGITUD", "LATITUDE", "TWAY_ID", "MILEPT", "RAIL",
                         "ARR_HOUR", "ARR_MIN", "HOSP_HR", "HOSP_MN"]

params = dict(objective="binary:logistic", eta=0.03, max_depth=10, silent=1, 
            subsample=0.9, colsample_bytree=1)
num_rounds = 500


def modifyPreds(preds):
    for i, p in enumerate(preds):
        p = preds[i]
        if p < 0: preds[i] = 0
        if p > 1: preds[i] = 1
    return preds

def validate():
    converters = dict(DRUNK_DR=convertDD, RAIL=convertRAIL, TWAY_ID=convertTWAYID)
    t1 = time.clock()
    acc_train_df = pandas.read_csv('accident_train.csv', converters=converters)
    t2 = time.clock()
    print "read accident_train.csv in {} seconds".format(t2 - t1)
    acc_train_df = acc_train_df.fillna(0)
    t3 = time.clock()
    print "filled NAs in {} secs".format(t3 - t2)

    veh_train_df = pandas.read_csv('vehicle_train.csv', dtype=dict(MCARR_ID=numpy.dtype(str)))
    t4 = time.clock()
    print "read vehicle_train.csv in {} secs".format(t4 - t3)
    veh_train_df = veh_train_df.fillna(0)
    t5 = time.clock()
    print "filled NAs in {} secs".format(t5 - t4)
    veh_train_df = veh_train_df[['ID', 'HARM_EV', 'HIT_RUN', 'PREV_DWI', 'TRAV_SP']]
    t6 = time.clock()
    print "sliced irrelevant columns in {} secs".format(t6 - t5)
    print "CSVs read in"

    new_cols = ['N_INV', 'ANY_PREV_DWI', 'MAX_TRAV_SP', 'ANY_HIT_RUN', 'HARM_EV']
    for new_col in new_cols:
        acc_train_df[new_col] = [0] * len(acc_train_df['ID'])
    ids = veh_train_df['ID'].unique()
    for i, id_ in enumerate(ids):
        if i % 1000 == 0:
            t7 = time.clock()
            print "{0:.3f} % {1:.1f} seconds elapsed".format(i / float(len(ids)), t7 - t6)
        this_id = acc_train_df[acc_train_df['ID'] == id_]
        vehicles = veh_train_df[veh_train_df['ID'] == id_]
        num_involved = len(vehicles)
        prev_dwi = sum([pd for pd in vehicles['PREV_DWI'] if pd != -1])
        max_trav_speed = max(vehicles['TRAV_SP'])
        any_hit_run = any([hr for hr in vehicles['HIT_RUN'] if hr != -1])
        any_hit_run = 1 if any_hit_run else 0
        harm_ev = vehicles['HARM_EV'].get_values()[0] # hopefully they're all the same for the same accident
        acc_train_df.set_value(this_id.index, 'N_INV', num_involved)
        acc_train_df.set_value(this_id.index, 'ANY_PREV_DWI', prev_dwi)
        acc_train_df.set_value(this_id.index, 'MAX_TRAV_SP', max_trav_speed)
        acc_train_df.set_value(this_id.index, 'ANY_HIT_RUN', any_hit_run)
        acc_train_df.set_value(this_id.index, 'HARM_EV', harm_ev)

    columns = list(acc_train_df.columns)
    for c in columns_to_remove:
        print c
        columns.remove(c)
    labels_train = acc_train_df['DRUNK_DR'].get_values()
    data_train = acc_train_df[columns]
    total_data = preprocess(data_train)

    itrainX, valX, itrainY, valY = cross_validation.train_test_split(total_data, 
            labels_train, test_size=0.4)

    xgtrain = xgboost.DMatrix(itrainX, label=itrainY)
    xgtest = xgboost.DMatrix(valX, label=valY)

    watchlist = [(xgtrain, 'train'), (xgtest, 'validation')]
    bst = xgboost.train(params, xgtrain, num_rounds, watchlist)
    preds = bst.predict(xgtest)
    print preds
    preds = modifyPreds(preds)
    print "AUROC: {}".format(roc_auc_score(valY, preds))
    
def predict():
    converters = dict(DRUNK_DR=convertDD, RAIL=convertRAIL, TWAY_ID=convertTWAYID)
    acc_train_df = pandas.read_csv('accident_train.csv', converters=converters)
    acc_train_df = acc_train_df.fillna(0)
    acc_test_df = pandas.read_csv('accident_test.csv', converters=converters)
    acc_test_df = acc_test_df.fillna(0)
    ids = acc_test_df['ID'].get_values()
    print "CSVs read in"

    columns = list(acc_train_df.columns)
    for c in columns_to_remove:
        print c
        columns.remove(c)
    columns.remove("YEAR") # test data doesn't have this key for some reason
    labels = acc_train_df['DRUNK_DR'].get_values()
    data_train = acc_train_df[columns]
    acc_test_df = acc_test_df[columns]

    xgtrain = xgboost.DMatrix(data_train, label=labels)
    xgtest = xgboost.DMatrix(acc_test_df)

    watchlist = [(xgtrain, 'train')]
    bst = xgboost.train(params, xgtrain, num_rounds, watchlist)
    preds = modifyPreds(bst.predict(xgtest))
    
    with open('submission.csv', 'w') as f:
        f.write("ID,DRUNK_DR\n")
        for i, id_ in enumerate(ids):
            f.write("{},{}\n".format(id_, preds[i]))
            
if __name__ == "__main__":
    if len(sys.argv) < 2:
        action = raw_input("predict (p) or validate (v): ")
    else:
        action = sys.argv[1]
    if action == 'p':
        print "Generating predictions"
        predict() 
    else:
        print "Validating model"
        validate()
