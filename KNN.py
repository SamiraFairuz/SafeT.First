#Imports
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
import pickle
import geopy.distance

def savemodel(object,filename='object_to_store.obj'):
	#store this object file 
    with open(filename, 'wb') as file:
        pickle.dump(object, file)

def loadmodel(filename):
	with open(filename, 'rb') as file:
		obj = pickle.load(file)
	return obj

def generateXY(filtered_df):
	X = filtered_df[['Longitude','Latitude']].to_numpy()
	Y = df[['OFFENCE']].loc[filtered_df.index].to_numpy()
	return X,Y

def filter_df(geo, range_in_km):
	Geo_feature = ['Latitude','Longitude']
	geo_data = df[Geo_feature]
	filtered_df = calculate_distances(geo, geo_data, range_in_km)
	return filtered_df

def group_by_geo():
	res2 = df.groupby(["Longitude", "Latitude"]).size().reset_index(name="Occurences").sort_values(by=['Occurences'],ascending=False)
	return res2

def Kmeans(k, df_var):
	kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto")
	kmeans.fit(df_var)
	return kmeans	

def Kmeans_predict(kmeans, new_df):
	predict = kmeans.predict(new_df)
	label = kmeans.labels_
	centroids = kmeans.cluster_centers_
	return predict, label, centroids

def get4ptrs(centroid,delta):
  #top left
  topleft = (centroid[0]-delta,centroid[1]+delta)
  topright = (centroid[0]+delta,centroid[1]+delta)
  downleft = (centroid[0]-delta,centroid[1]-delta)
  downright= (centroid[0]+delta,centroid[1]-delta)
  return [topleft,topright,downleft,downright]

def calculate_distances(centroid, dataframe, threshold):
    # Calculate distances
    dataframe['Distance'] = dataframe.apply(lambda row: geopy.distance.distance(centroid, (row['Latitude'], row['Longitude'])).km, axis=1)
    # Filter dataframe based on distance threshold
    filtered_dataframe = dataframe[dataframe['Distance'] < threshold]
    return filtered_dataframe

def KNN(k,X,Y):
	model = KNeighborsClassifier(n_neighbors=3,weights='distance')
	model.fit(X,Y)
	return model

def KNN_predict(model, newpoint):
	distances, indices = model.kneighbors([newpoint])
	return distances, indices 

#main script
if __name__ == '__main__':
    df = pd.read_csv('.\Dataset_June_8.csv')
    # build the dictionary:
    dict_offense = {'Assault': 9, 'Break and Enter': 1, 'Robbery': 5, 'Auto Theft': 1, 'Homicide': 15, 'Theft Over': 10, 'Pedestrian Collision': 6}
    # replace the column values:
    df2=df.replace({"OFFENCE": dict_offense})

    geo_loc = (43.609206, -79.514755)
    #177-137 Horner Ave, Etobicoke, ON

    filtered_df = filter_df(geo_loc, 5)
    x, y = generateXY(filtered_df)

    #Trained your KNN
    knn = KNN(100,x,y)
    newpoint = (43.616875, -79.519073)
    distances, indices = KNN_predict(knn,newpoint) 
    savemodel(knn,'100nbrs.obj')
    #reload KNN:
    KNN100 = loadmodel('100nbrs.obj')
    print("success")
	