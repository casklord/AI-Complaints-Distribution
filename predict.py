import pandas as pd
import numpy as np
import math
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
#from openpyxl import load_workbook


def clean_data():
	filename = "SAM Data.csv"
	build_dp = ["redacted for privacy"] #previous values removed
	design_dp = ["redacted for privacy"] #previous values removed
	df = pd.read_csv(filename)
	num_rows = len(df.index)
	x = np.zeros((num_rows,6 + 21 + 27))
	y = np.zeros((num_rows,6))

	for i in range(0, num_rows):
		#read distribution and complaints number
		for j in range(0,6):
			y[i][j] = int(df.iloc[i][j+9])
		#read technology population numbers
		for j in range(0,6):
			if math.isnan(df.iloc[i][j+1]):
				x[i][j] = 0
			else:
				x[i][j] = int(df.iloc[i][j+1])
		#read in build dp name
		build_dp = df.iloc[i][7]
		for b in build:
			if b == build_dp:
				x[i][build.index(b)+6] = 1
				break
		#read in design dp name
		design_dp = df.iloc[i][8]
		for d in design:
			if d == design_dp:
				x[i][design.index(d)+6+21] = 1
				break

	z = []
	for i in range(0,num_rows):
		record = np.append(x[i],y[i])
		z.append(record)

	newDF = pd.DataFrame(z)
	newDF.to_csv("SAM Data Cleaned.csv", index=False)
	return

def clean_data2():
	filename = "SAM Data2.csv"
	build_dp = ["redacted for privacy"] #previous values removed
	design_dp = ["redacted for privacy"] #previous values removed
	df = pd.read_csv(filename)
	num_rows = len(df.index)
	x = np.zeros((num_rows,6 + 21 + 27))

	for i in range(0, num_rows):
		#read technology population numbers
		for j in range(0,6):
			#print(df.iloc[i][j+1])
			if math.isnan(df.iloc[i][j+1]):
				x[i][j] = 0
			else:
				x[i][j] = int(df.iloc[i][j+1])
		#read in build dp name
		build_dp = df.iloc[i][7]
		for b in build:
			if b == build_dp:
				x[i][build.index(b)+6] = 1
				break
		#read in design dp name
		design_dp = df.iloc[i][8]
		for d in design:
			if d == design_dp:
				x[i][design.index(d)+6+21] = 1
				break

	z = []
	for i in range(0,num_rows):
		record = x[i]
		z.append(record)

	newDF = pd.DataFrame(z)
	newDF.to_csv("SAM test Cleaned.csv", index=False)
	return

def train_AI_count():
	filename = "SAM Data Cleaned.csv"
	df_X = pd.read_csv(filename, usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53])
	df_y = pd.read_csv(filename, usecols=[54,55,56,57,58,59])
	X = df_X.to_numpy()
	y = df_y.to_numpy()
	y = [item[5] for item in y]
	clf = MLPRegressor(activation='relu', alpha=1e-05, batch_size='auto', beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08,
		hidden_layer_sizes=(20, 10), learning_rate='constant', learning_rate_init=0.001, max_iter=300, momentum=0.9, nesterovs_momentum=True, 
		power_t=0.5, random_state=1, shuffle=True, solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False, warm_start=False)
	
	#loop
	for i in range(0,5):
		train_X, validate_X, train_Y, validate_Y = train_test_split(X, y, test_size=0.2, shuffle=True)
		#print(len(train_X))
		#print(len(validate_X))
		clf.fit(train_X, train_Y)   
		predictions = clf.predict(validate_X)
		score = r2_score(predictions, validate_Y)
		print("Iteration " + str(i) + " score is: " + str(score))

	#print(clf.get_params())
	filename = "SAM test Cleaned.csv"
	df_X = pd.read_csv(filename, usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53])
	X = df_X.to_numpy()
	predictions = clf.predict(X)
	z = []
	for i in range(0,len(X)):
		record = predictions[i]
		z.append(record)
	newDF = pd.DataFrame(z)
	newDF.to_csv("Predictions.csv", index=False)
	return

def train_AI_dist():
	filename = "SAM Data Cleaned.csv"
	df_X = pd.read_csv(filename, usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53])
	df_y = pd.read_csv(filename, usecols=[54,55,56,57,58,59])
	X = df_X.to_numpy()
	y = df_y.to_numpy()
	y = [item[0:5] for item in y]
	clf = MLPRegressor(activation='relu', alpha=1e-04, batch_size='auto', beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08,
		hidden_layer_sizes=(15, 8), learning_rate='constant', learning_rate_init=0.002, max_iter=200, momentum=0.9, nesterovs_momentum=True, 
		power_t=0.5, random_state=1, shuffle=True, solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False, warm_start=False)
	#loop
	#TODO change tech type to percentages
	scores = []
	for i in range(0,20):
		train_X, validate_X, train_Y, validate_Y = train_test_split(X, y, test_size=0.2, shuffle=True)
		clf.fit(train_X, train_Y)   
		predictions = clf.predict(validate_X)
		
		score = r2_score(predictions, validate_Y)
		scores.append(score)
		print("Iteration " + str(i) + " score is = " + str(score))
		#print("Average score = " + str(sum(scores)/float(len(scores))))
	#print(clf.get_params())

	predictions = clf.predict(X)
	z = []
	for i in range(0,len(X)):
		record = np.append(predictions[i],y[i])
		z.append(record)
	newDF = pd.DataFrame(z)
	newDF.to_csv("Predictions dist.csv", index=False)
	return

clean_data2()
clean_data()

train_AI_dist()
train_AI_count()