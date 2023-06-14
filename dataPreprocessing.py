#Imports
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt


df = pd.read_csv('.\Temp_Data_June_7.csv')

#print(f"Robbery df type: {type(df)}")
#print(f"Robbery shape: {df.shape}")

dict_offense = {'Assault': 9, 'Break and Enter': 1, 'Robbery': 5, 'Auto Theft': 1, 'Homicide': 15, 'Theft Over': 10}
df2=df.replace({"OFFENCE": dict_offense})
df2.drop('Unnamed: 0', inplace=True, axis=1)

#PCA
pca = PCA(2)
df_projected = pca.fit_transform(df2)
df_projected.shape

#Elbow Method
'''wcss=[]
X=df2.iloc[:, [0,1]].values
for i in range(5,20): 
     kmeans = KMeans(n_clusters=i, init ='k-means++', max_iter=300,  n_init=10,random_state=0 )
     kmeans.fit(X)
     wcss.append(kmeans.inertia_)

plt.plot(range(5,20),wcss)
plt.title('The Elbow Method Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()'''

#K-means on entire dataset
'''kmeans = KMeans(n_clusters=15, random_state=0, n_init="auto")
kmeans.fit(df2)
predicted_crime = kmeans.predict(df2)
#Getting unique labels
label = kmeans.labels_
u_labels = np.unique(kmeans.labels_)'''
#print(kmeans.cluster_centers_)

#Separating Lat/Long
Geo_feature = ['Latitude','Longitude']
geo_data = df[Geo_feature]

#Separating First 20,000 data
df4 = df2.iloc[0:20000,:]
geo_data2 = geo_data.iloc[0:20000,:]

#Sort the data according to Long/Lat or use filter function using some ranges 
#Find offence values for each cluster

#PCA and Kmeans on df4
pca = PCA(2)
df_projected4 = pca.fit_transform(df4)
kmeans2 = KMeans(n_clusters=100, random_state=0, n_init="auto")
kmeans2.fit(df4)
predicted_crime4 = kmeans2.predict(df4)
#Getting unique labels
label2 = kmeans2.labels_
u_labels2 = np.unique(kmeans2.labels_)
#np.savetxt("label.csv", label2, delimiter=",")
df_label = pd.DataFrame(label2)

#Get the counts for each unique value
counts = df_label.value_counts()
pd.DataFrame(counts).to_csv("./label_value_counts.csv")
#print(counts)
df_label.to_csv("./labels.csv")
#print(label2)

#Getting the Centroids
centroids = kmeans2.cluster_centers_

#plotting the results:
'''for i in u_labels2:
    plt.scatter(df_projected4[label2 == i , 0] , df_projected4[label2 == i , 1] , label = i)
plt.scatter(centroids[:,0] , centroids[:,1] , color = 'k')
plt.legend()
plt.show()'''

#Creating Dataframe
#print(centroids)
df_centroid = pd.DataFrame(centroids,columns=['Latitude','Longitude','OFFENCE','DAYS_SINCE'])
df_centroid.to_csv('./centroids.csv')
#print(df_centroid.columns)

#Finding Minimum and Maximum Lat/Long values
min_series = df_centroid.min()
lat_min = min_series[0]
long_min = min_series[1]
#print(lat_min)
#print(long_min)

max_series = df_centroid.max()
lat_max = max_series[0]
long_max = max_series[1]
#print(lat_max)
#print(long_max)
#########################################################################################################
#Kmeans on Geo data
kmeans3 = KMeans(n_clusters=100, random_state=0, n_init="auto")
kmeans3.fit(geo_data2)
predicted_crime5 = kmeans3.predict(geo_data2)
#Getting unique labels
label3 = kmeans3.labels_
u_labels3 = np.unique(kmeans3.labels_)

df_label2 = pd.DataFrame(label3)
#Get the counts for each unique value
counts2 = df_label2.value_counts()
pd.DataFrame(counts2).to_csv("./geo_label_value_counts.csv")
#print(counts2)

#Getting the Centroids
centroids2 = kmeans3.cluster_centers_
df_centroid2 = pd.DataFrame(centroids2,columns=['Latitude','Longitude'])
df_centroid2.to_csv('./centroids_geo.csv')

#Add OFFENCE column to labels data
print(df4.columns)
feature_add = ['OFFENCE', 'DAYS_SINCE']
temp_df = df4[feature_add]
#print(temp_df)
df_label_crime = pd.concat([df_label2,temp_df], axis=1)
#label_crime.reshape(20000,2)
#label_crime = np.concatenate((label3,temp),axis=1)
#df_label_crime = pd.DataFrame(label_crime)

df_label_crime.to_csv("./labels_geo_with_crime.csv")
