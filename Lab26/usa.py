#Hierarchical clustering
#do hierarchical clustering on USArrests dataset
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial import distance_matrix
from sklearn.preprocessing import StandardScaler
from statsmodels.datasets import get_rdataset
import scipy.cluster.hierarchy as sch


#load the dataset
USArrests = get_rdataset('USArrests').data
labels = USArrests.index.to_list()
# print(labels)
#standardize the data

scaler = StandardScaler()
data_scaled = scaler.fit_transform(USArrests)
# print(data_scaled)

#a)
#perform hierarchical clustering on the scaled data by states

hc_states = sch.linkage(data_scaled,method='complete',metric = 'euclidean')
print(hc_states)
print(hc_states.shape)

#clusters
max_cluster = 3
clusters = sch.fcluster(hc_states, t=max_cluster, criterion='maxclust' )
print(clusters)

#print the states belonging to each cluster
cluster_df = pd.DataFrame({'state_name': USArrests.index, 'cluster':clusters})
print(cluster_df)

##### Print the name of states in each cluster
for i in range(1, max_cluster + 1):
    states_in_cluster = cluster_df[cluster_df['cluster'] == i]['state_name'].values
    print(f"\nCluster {i} states:")
    print(states_in_cluster)

#b)
cut_height = hc_states[-max_cluster+1,2] - 0.0001

hc_den = sch.dendrogram(hc_states,
                        link_color_func= lambda x: 'black',
                        labels = labels,
                        )
plt.title("Hierarchical clustering USArrests data")
plt.xlabel("State_names")
plt.ylabel("cluster distances")
plt.axhline(y=cut_height,linestyle='--',color='red')
plt.show()



##C) correlation based matrix distance

hc_states = sch.linkage(data_scaled,method='complete',metric = 'correlation')
print(hc_states.shape)