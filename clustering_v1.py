# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 17:24:55 2018
@author: murali.sai
"""
import warnings, pandas as pd, os, numpy as np
from sklearn.mixture import GMM
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=DeprecationWarning)

#Read Data
data = pd.read_csv(os.path.join(os.pardir,'data/final_sheet.csv'));
data = data.iloc[data.index[data['ingredient_not_found']=='SUCCESS'],:]
data.drop(['ingredient_not_found','Error','item_link','quantity'], axis=1, inplace=True);

#Process foodData
foodNames = {};
foodFeatures = np.array(data.iloc[:,2:].astype(float));
foodFeaturesNames = data.columns.values.tolist()[2:];
for ind, item in enumerate(data['recipe_name']): 
    foodNames[ind] = item;
    foodFeatures[ind,:]/=(0.01*float(data.iloc[ind,1]));
del item, ind;

#Find optimal number of clusters
n_components_1 = [1, 2, 4, 8, 16, 32, 64, 128, 256];
AIC, BIC = [], [];
for n in n_components_1:
    gmm = GMM(n_components=n, n_iter=1000)
    gmm.fit(foodFeatures)
    AIC+= [gmm.aic(foodFeatures)];
    BIC+= [gmm.bic(foodFeatures)];
plt.figure()
plt.plot(n_components_1, AIC, '-*r', label='AIC')
plt.plot(n_components_1, BIC, '-*b', label='BIC')
plt.legend(loc=0)
plt.xlabel('n_components_1')
plt.ylabel('AIC / BIC')
del n, n_components_1

n_components_2 = [12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32];
AIC, BIC = [], [];
for n in n_components_2:
    gmm = GMM(n_components=n, n_iter=1000)
    gmm.fit(foodFeatures)
    AIC+= [gmm.aic(foodFeatures)];
    BIC+= [gmm.bic(foodFeatures)];
plt.figure()
plt.plot(n_components_2, AIC, '-*r', label='AIC')
plt.plot(n_components_2, BIC, '-*b', label='BIC')
plt.legend(loc=0)
plt.xlabel('n_components_1')
plt.ylabel('AIC / BIC')
del n, n_components_2

# a good cluster number
n_components = 30;
gmm = GMM(n_components=n_components, n_iter=1000);
clusters = gmm.fit_predict(foodFeatures);

# Save all elements of same cluster to individual csv files
foodClusters = {};
for i in range(n_components):
    foodClusters[i] = [];
for ind, clusterID in enumerate(clusters):
    foodDetails = []; 
    foodDetails+=[foodNames[ind]];
    for i in list(foodFeatures[ind,:]):
        foodDetails+=[i];
    foodClusters[clusterID]+=[foodDetails]
del i, ind, clusterID, foodDetails
foodDF = [];
for i in range(n_components):
    try:
        df = pd.DataFrame(foodClusters[i]);
        foodDF = pd.concat([foodDF,df]);
    except:
        foodDF = pd.DataFrame(foodClusters[i])
    foodDF    
    columns = ['recipe_name']+foodFeaturesNames
    df.to_csv(os.path.join(os.pardir,'data/gmm_cluster_'+str(i)+'.csv'));
del i


#labels = gmm.predict(X)
#plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis');