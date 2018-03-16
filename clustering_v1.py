# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 17:24:55 2018
@author: murali.sai
"""
# ##############################################################################
# NOTES
# Ideally speaking, we want to have tags like 'high/low' for every attribute
# Mathematically, we have 2^11 possibilities. That is 2048.
# But we want to restrict ourselves to the popular (dense) clusters
# ##############################################################################

# Imports
import warnings, pandas as pd, os, numpy as np
from numpy.linalg import norm
from sklearn.mixture import GMM
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=DeprecationWarning)
#Definitions
thresholds_high = {'energy_gm':300, 'protein_gm':40, 'fat_gm':25, 'carbs_gm':50,
                   'total_dietary_fibre_gm':30, 'trans_fat_gm':0,
                   'cholesterol_mg':0, 'added_sugar_gm':0};
                   #'sugar_gm':np.inf, 'sodium_mg':np.inf, 'total_saturated_fat_gm':np.inf,
thresholds_low = {'fat_gm':10, 'energy_gm':100, 'protein_gm':15, 'carbs_gm':15,
                  'total_dietary_fibre_gm':10};
data = []; attributeNames = [];
foodNames = {}; foodVectors = [];


def draw_aic_bic(foodVectors, n_components):
    AIC, BIC = [], [];
    for n in n_components:
        gmm = GMM(n_components=n, n_iter=1000)
        gmm.fit(foodVectors)
        AIC+= [gmm.aic(foodVectors)];
        BIC+= [gmm.bic(foodVectors)];
    plt.figure()
    plt.plot(n_components, AIC, '-*r', label='AIC')
    plt.plot(n_components, BIC, '-*b', label='BIC')
    plt.legend(loc=0)
    plt.xlabel('n_components_1')
    plt.ylabel('AIC / BIC')
    return AIC, BIC

if __name__=="__main__":
    # Read Data final_sheet.csv and check if all required columns exist
    data = pd.read_csv(os.path.join(os.pardir,'data/clustering/originals/norm_data.txt'), sep='\t')
    for kys in thresholds_high.keys():
        if kys not in data.columns.values.tolist():
            print(kys, 'NOT FOUND! IN DATA FILE')
    for kys in thresholds_low.keys():
        if kys not in data.columns.values.tolist():
            print(kys, 'NOT FOUND! IN DATA FILE')
    del kys
    #data = pd.read_csv(os.path.join(os.pardir,'data/clustering/originals/<sheet_name>.csv'),encoding='latin-1');

    # Process the read data
    data = data.iloc[data.index[data['ingredient_not_found']=='SUCCESS'],:]
    data.drop(['ingredient_not_found','Error','item_link','quantity'], axis=1, inplace=True);
    data.drop(['0','0.1','0.2'], axis=1, inplace=True);
    foodNames = {};
    foodVectors = np.array(data.iloc[:,2:].astype(float));
    attributeNames = data.columns.values.tolist()[2:];
    for ind, item in enumerate(data['recipe_name']):
        foodNames[ind] = item;
        foodVectors[ind,:]/=(0.01*float(data.iloc[ind,1]));
    del item, ind;

    # Draw AIC, BIC
    #n_components = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
    #AIC, BIC = draw_aic_bic(foodVectors, n_components);
    #n_components = [22, 24, 26, 28, 30, 32, 34, 36, 38, 40];
    #AIC, BIC = draw_aic_bic(foodVectors, n_components);
    #del AIC, BIC, n_components;

    # Select a good cluster number
    n_components = 32;
    gmm = GMM(n_components=n_components, n_iter=1000);
    clusters_argmax = gmm.fit_predict(foodVectors);
    clusters_scores = gmm.score_samples(foodVectors);
    cluster_centers = gmm.means_;
    clusters_order_sorted=[];
     # Add Cluster IDs to each food Item
    data_clustered = data.copy();
    data_clustered['clusterID'] = clusters_argmax;

    # Give tags to each cluster
    tags = pd.DataFrame([], columns=[name+'_tag' for name in attributeNames], index=range(cluster_centers.shape[0]))
    tags['clusterID'] = range(cluster_centers.shape[0]);
    for ind, thisCenter in enumerate(cluster_centers):
        for attr_ind, attr_value in enumerate(thisCenter):
            attr_name = attributeNames[attr_ind];
            if attr_name in thresholds_high.keys() and attr_value>thresholds_high[attr_name]:
                tags.loc[ind,attr_name+'_tag'] = 'HIGH';
            if attr_name in thresholds_low.keys() and attr_value<thresholds_low[attr_name]:
                tags.loc[ind,attr_name+'_tag'] = 'LOW';
    #unique_tags = []; unique_tags = tags.drop_duplicates();
    del ind, thisCenter, attr_ind, attr_value, attr_name

    # Sort cluster indices in des_order based on the
    # distance of cluster center from origin
    cluster_centers_dists = [];
    for row in cluster_centers:
        cluster_centers_dists+=[-norm(row)]
    clusters_order_sorted = np.argsort(cluster_centers_dists);
    del row, cluster_centers_dists;
    # Merge data and tags and sort
    data_clustered = pd.merge(data_clustered, tags, on='clusterID', how='inner')
    data_clustered['clusterID_cat'] = pd.Categorical(
        data_clustered['clusterID'],
        categories=clusters_order_sorted,
        ordered=True
    )
    data_clustered.sort_values('clusterID_cat', inplace=True)
    data_clustered.drop(['clusterID_cat'], axis=1, inplace=True);

    # Save data files
    data_clustered.to_csv(os.path.join(os.pardir,'data/clustering/clustered/data.csv'),index=False);
    tags.to_csv(os.path.join(os.pardir,'data/clustering/clustered/tags_high_low.csv'),index=False)






# =============================================================================
# fooddata_clustered = [];
# for i in range(n_components):
#     try:
#         data_clustered = pd.DataFrame(foodClusters[i]);
#         fooddata_clustered = pd.concat([fooddata_clustered,data_clustered]);
#     except:
#         fooddata_clustered = pd.DataFrame(foodClusters[i])
#     fooddata_clustered
#     columns = ['recipe_name']+attributeNames
#     data_clustered.to_csv(os.path.join(os.pardir,'data/clustered/gmm_cluster_'+str(i)+'.csv'));
# del i
# =============================================================================
#==============================================================================
# foodClusters = {};
# foodClusters_all = [];
# for ind, clusterID in enumerate(clusters_argmax):
#     foodDetails = [];
#     foodDetails+=[clusterID];
#     foodDetails+=[foodNames[ind]];
#     for i in list(foodVectors[ind,:]):
#         foodDetails+=[i];
#     foodClusters[clusterID]+=[foodDetails];
#     foodClusters_all+=[foodDetails];
# data_clustered = pd.DataFrame(foodClusters_all, columns=['clusterID']+['name']+attributeNames);
# del foodClusters, foodClusters_all, i, ind, clusterID, foodDetails
#==============================================================================