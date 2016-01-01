import pandas, json

df = pandas.read_csv('train.csv')

with open('cats.json', 'r') as f:
    cats = json.load(f)

scores = []

for col in cats:
    unique_vals = sorted(df[col].unique())
    avg = 0
    for uv in unique_vals:
        above4 = len(df[(df[col] == uv) & (df['Response'] > 4)])
        below4 = len(df[(df[col] == uv) & (df['Response'] <= 4)])
        if above4 == 0 and below4 == 0:
            continue
        score = abs(0.5 - above4 / float(above4 + below4)) * 2
        avg += score
    scores.append([col, avg / float(len(unique_vals))])
    
scores.sort(key=lambda tup: tup[1])
print ', '.join(["'{}'".format(s[0]) for s in scores])
