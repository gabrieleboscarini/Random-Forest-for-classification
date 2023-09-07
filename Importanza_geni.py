
import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import roc_auc_score
import pandas as pd
from sklearn.ensemble import RandomForestClassifier



data = pd.read_csv('C:/Users/gabro/OneDrive/Desktop/Tesi2021/datasets/db_perugia_immunofenotipi.csv') #Dataset : db_perugia_immunofenotipi
data.drop(data.loc[data['Target']==2].index, inplace=True)


#Feature Selection
correlated_features = set()
correlation_matrix = data.corr()

for i in range(len(correlation_matrix .columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.95:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)

data.drop(labels=correlated_features, axis=1, inplace=True)

correlation_output = data.corr()['Target']

correlated_with_output = set()
for i in range(len(correlation_output)):
    if abs(correlation_output.iloc[i]) < 0.1:
         correlated_with_output.add(correlation_output.index[i])

data.drop(labels=correlated_with_output, axis=1, inplace=True)

all_features = [x for x in data.drop(['Patient', 'Target'], axis=1).columns]

X = data.drop(columns=['Target', 'Patient'], axis=1).to_numpy()
y = data['Target'].to_numpy()

rfc = RandomForestClassifier(n_estimators= 800, min_samples_split= 10, min_samples_leaf= 2, max_features= 'sqrt', max_depth= 30, bootstrap= False)
rfecv = RFECV(
              estimator=rfc,
              step=2,
              cv=StratifiedKFold(n_splits=4).split(X,y),
              scoring='accuracy',
              n_jobs=1,
              verbose=2)

rfecv.fit(X, y)

print('\n Optimal number of features: %d' % rfecv.n_features_)
sel_features = [f for f, s in zip(all_features, rfecv.support_) if s]
print('\n The selected features are {}:'.format(sel_features))

