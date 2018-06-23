#!/usr/bin/python

import sys
import pickle
import collections
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

####################### CLASSIFIERS ##########################
from sklearn.naive_bayes import GaussianNB                   #
from sklearn.ensemble import RandomForestClassifier          #
from sklearn.neighbors import KNeighborsClassifier           #
from sklearn.ensemble import GradientBoostingClassifier      # 
##############################################################


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
'''
all_features = ['salary', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options', 
				 'bonus', 'restricted_stock', 'shared_receipt_with_poi', 'restricted_stock_deferred', 
				 'total_stock_value', 'expenses', 'loan_advances', 'from_messages', 'other', 'from_this_person_to_poi', 
				 'poi', 'director_fees', 'deferred_income', 'long_term_incentive', 'email_address', 'from_poi_to_this_person'] 

'''

#Experiments:
#features_list = ['poi','salary','exercised_stock_options', 'bonus', 'deferred_income', 'long_term_incentive'] 
#features_list = ['poi','salary', 'expenses', 'loan_advances', 'from_this_person_to_poi', 'from_poi_to_this_person']
#features_list = ['poi','salary', 'shared_receipt_with_poi', 'restricted_stock_deferred', 'from_this_person_to_poi', 'from_poi_to_this_person']
#features_list = ['poi','salary', 'director_fees', 'long_term_incentive', 'from_this_person_to_poi', 'from_poi_to_this_person']
#features_list = ['poi','total_stock_value', 'expenses', 'loan_advances', 'from_this_person_to_poi', 'from_poi_to_this_person']

#Best with only original features:
#features_list = ['poi','salary','exercised_stock_options', 'bonus', 'from_this_person_to_poi', 'from_poi_to_this_person'] 

#Best with addition of new feature:
features_list = ['poi','salary','exercised_stock_options', 'bonus', 'from_this_person_to_poi', 'from_poi_to_this_person', 'restricted_stock_ratio'] 

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
	data_dict = pickle.load(data_file)

'''
import matplotlib.pyplot as plt 

data = collections.defaultdict(list)
names = []
for feature in features_list:
	for name in data_dict:
		if name != 'TOTAL': # Found from noticing huge outlier
			names.append(name)
			data[feature].append(data_dict[name][feature])


used = []
for financial_feature in financial_features:
	used.append(financial_feature)
	for other_feature in financial_features: 
		if other_feature not in used:
			plt.scatter(data[financial_feature], data[other_feature])
			plt.xlabel(financial_feature)
			plt.ylabel(other_feature)
			plt.show()
'''

### Task 2: Remove outliers
'''
for employee in data_dict:
	print(employee)
for employee in data_dict:
	print(data_dict[employee])
'''

data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0) ### Clearly not a person 
data_dict.pop('TOTAL', 0) ### Not applicable, so we should remove

### Task 3: Create new feature(s)

for employee in data_dict:
	if round(float(data_dict[employee]['restricted_stock']) / float(data_dict[employee]['total_stock_value']), 2) > 0:
		data_dict[employee]['restricted_stock_ratio'] = round(float(data_dict[employee]['restricted_stock']) / float(data_dict[employee]['total_stock_value']), 2)
	else: 
		data_dict[employee]['restricted_stock_ratio'] = 0
### ^^^Add'restricted_stock_ratio' feature to the data dictionary by calculating the ratio of restricted_stock to total_stock_value^^^

### Save final dataset:
final_dataset = data_dict

### Scale features with MinMaxScaler:
labels, features = targetFeatureSplit(featureFormat(final_dataset, features_list, sort_keys=True))
features = preprocessing.MinMaxScaler().fit_transform(features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

### Initialize classifier from options (comment out all but one):

#clf = GaussianNB()
#clf = RandomForestClassifier(n_estimators=10) 
clf = KNeighborsClassifier(n_neighbors=6, weights='distance', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)
#clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
#features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, final_dataset, features_list)