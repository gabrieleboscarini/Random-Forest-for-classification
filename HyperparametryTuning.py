import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import RandomizedSearchCV


data = pd.read_csv('C:/Users/gabro/Desktop/Tesi2021/datasets/db_perugia_immunofenotipi.csv')
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

train_features, test_features, train_labels, test_labels = train_test_split(
    data.drop(labels=['Target','Patient'], axis=1),
    data['Target'],
    test_size=0.25,
    random_state=42)

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf = RandomForestClassifier()

# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 2, verbose=2, random_state=42, n_jobs = -1)

rf_random.fit(train_features, train_labels)

print(rf_random.best_params_)



















