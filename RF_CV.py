import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedKFold, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
from sklearn import preprocessing
import matplotlib.pyplot as plt



data = pd.read_csv('C:/Users/gabro/OneDrive/Desktop/Tesi2021/datasets/db_perugia_immunofenotipi.csv') #Dataset : db_perugia_immunofenotipi

data.drop(data.loc[data['Target']==2].index, inplace=True)

y = data['Target']
data.drop('Patient',inplace=True,axis =1)

scaler = preprocessing.MinMaxScaler()
data = pd.DataFrame(scaler.fit_transform(data), columns= data.columns)
data.head()

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


feature_list = list(data.columns)
feature_list.remove('Target')
data.drop('Target',inplace=True,axis =1)

#data=data[['ATF2', 'ATG10', 'BAX', 'BST1', 'C2', 'C6', 'CASP1', 'CCL5', 'CD36', 'CD4', 'CD58', 'CFD', 'CXCL13', 'IL11RA', 'IL16', 'IL3RA', 'IRF5', 'IRF8', 'JAK2', 'KIR3DL1', 'KIT', 'MAPK3', 'MICA', 'NCF4', 'PIK3CD', 'PIK3CG', 'REPS1', 'SPP1', 'STAT1', 'TGFB1', 'TIRAP', 'TLR3', 'TNFSF10']]

#Creating the random forest
rf = RandomForestClassifier(n_estimators= 1000, min_samples_split= 2, min_samples_leaf= 1, max_features= 'auto', max_depth= 50, bootstrap= False)

#Cross Validation process
kf = StratifiedKFold(n_splits=4,shuffle=False, random_state=None)

accuracies = []


for train_index, test_index in kf.split(data, y):
    X_train = data.values[train_index]
    X_test = data.values[test_index]
    y_train = y.values[train_index]
    y_test = y.values[test_index]
    rf.fit(X_train, y_train)
    conf_mat = confusion_matrix(y_test,rf.predict(X_test))
    plt.figure()
    hm = sns.heatmap(conf_mat, annot=True)
    hm.set(xlabel = "predicted features", ylabel = "True features")
    accuracies.append(accuracy_score(y_test, rf.predict(X_test)))
    print(accuracy_score(y_test, rf.predict(X_test)))

#Printing the average of accuracies of each fold
print(np.mean(accuracies))

#Creating a list of features with the relative importance in classification process
importance = rf.feature_importances_

for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))

# plot feature importance
plt.figure()
plt.title("Features Importance Graph")
plt.xlabel('Feature index')
plt.ylabel('Feature importance')
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


feature_importance = np.array(importance)
feature_names = np.array(feature_list)

data={'feature_names':feature_names,'feature_importance':feature_importance}
fi_df = pd.DataFrame(data)

#Sort the DataFrame in order decreasing feature importance
fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

#Define size of bar plot
plt.figure(figsize=(10,8))
#Plot Searborn bar chart
sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
#Add chart labels
plt.title('Random Forest' + 'FEATURE IMPORTANCE')
plt.xlabel('FEATURE IMPORTANCE')
plt.ylabel('FEATURE NAMES')



