import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import time 
from sklearn import preprocessing 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.base import ClusterMixin
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, Birch
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NerestNeighbors
from sklearn import metrics
from yellowbrick.cluster import SilhouetteVisualizer, KElbowVisualizer
from yellowbrick.cluster import KElbowVisualizer
import seaborn as sns
import scipy.cluster.hierarchy as sch


class GMClusters(GaussianMixture, ClusterMixin):
    '''
    Elbow wrapper class for GMM: https://stackoverflow.com/questions/58648969/the-supplied-model-is-not-a-clustering-estimator-in-yellowbrick
    '''

    def fit(self, X):
        super().fit(X)
        self.labels_ = self.predict(X)
        return self

    def get_params(self, **kwargs):
        output = super().get_params(**kwargs)
        output["n_clusters"] = output.get("n_components", None)
        return output

    def set_params(self, **kwargs):
        kwargs["n_components"] = kwargs.pop("n_clusters", None)
        return super().set_params(**kwargs)


def plot_hist(df):
    hist = df.hist(layout=(7,2),figsize=(10, 10))
    plt.title("Hisogram plot", size=5, weight='bold')
    plt.tight_layout()
    plt.show()

def plot_box(df, col):
    # fig = plt.figure()
    # fig.add_subplot(1, 3, 2)
    fig = df.boxplot(column=col)
    fig.set_title('box plot')
    plt.tight_layout()
    # fig.set_xlabel('danceability')
    # fig.set_ylabel('value')
    plt.show()
    
def sns_box(df):
    plt.figure(figsize=(15,6))
    sns.boxplot(df)
    plt.show()
    
def plot_dist(df):
    '''plot the distribution of a given column in a dataframe'''
    plt.figure(figsize=(12, 6))
    sns.distplot(df['speechiness'], bins=40)
    plt.show()   

def plot_bar(x, y, title):
    '''
    input two lists
    '''
    plt.barh(x, y, height=0.5)
    #plt.xlabel('genre', fontsize=8)
    plt.title(f'Cluster {title}')
    plt.tight_layout()
    plt.show()

def preprocess(df):
    df = df.drop(axis = 1, columns = ['id', 'type', 'uri', 'track_href', 'analysis_url', 'song_name', 'Unnamed: 0', 'title'])
    duration_min = df['duration_ms'] / 60000.0
    df.drop(axis=1, columns='duration_ms', inplace=True)
    genre = df['genre']
    df.drop(axis = 1, columns=['genre'], inplace=True)
    df['duration_min'] = duration_min
    
    # encoding = preprocessing.LabelEncoder()
    # encoding.fit(df['genre'])
    # df['genre'] = encoding.transform(df['genre'])
    
    df_scaled = StandardScaler().fit_transform(df)
    df = pd.DataFrame(df_scaled)
    df.set_axis(['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
       'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
       'time_signature', 'duration_min'], axis=1, inplace=True)
    # df = remove_noise(df)
    # print(df.skew().sort_values(ascending=False)) 
    # df['acousticness'] = df['acousticness'].apply(lambda x: x + 1)
    # df['liveness'] = df['liveness'].apply(lambda x: x + 2)
    # df['speechiness'] = df['speechiness'].apply(lambda x: x + 1)
    # df['time_signature'] = df['time_signature'].apply(lambda x: x + 12)
    # df [['acousticness', 'liveness', 'speechiness', 'time_signature']]= df[['acousticness', 'liveness', 'speechiness', 'time_signature']].apply(lambda x: np.log(x)) 
    
    df['genre'] = genre
    df.dropna(axis=0, inplace=True)

    duration_min = df['duration_min']
    genre = df['genre']
    df.drop(axis = 1, columns=['duration_min', 'genre'], inplace=True)
    
    # df.hist()
    # plt.tight_layout()
    # plt.show()

    # df.drop(axis = 1, columns=['genre'], inplace=True)
    # print(df.isnull().sum())
    # print(df.describe(include='all'))

    return df, duration_min, genre

def remove_noise(df):
    '''
    remove the noise of the certain given column
    '''
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    return df[(df >= lower_bound) & (df <= upper_bound)]

def elbow(df):
    # kmeans_kwargs = {
    #     "init": "random",
    #     "n_init":10,
    #     "random_state": 1,
    # }

    # #create list to hold SSE values for each k
    # sse = []
    # for k in range(1, 11):
    #     kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    #     kmeans.fit(df)
    #     sse.append(kmeans.inertia_)

    # #visualize results
    # plt.plot(range(1, 11), sse, marker='o')
    # plt.xticks(range(1, 11))
    # plt.xlabel("Number of Clusters")
    # plt.ylabel("SSE")
    # plt.show()
    # ----------------------------------------------
    model = KMeans()
    # model = AgglomerativeClustering()
    '''Birch'''
    # model = Birch(n_clusters=None)
    visualizer = KElbowVisualizer(model, k=(2, 10))
    
    '''GMM'''
    # visualizer = KElbowVisualizer(GMClusters(), k=(2, 10), force_model = True)
    visualizer.fit(df)
    visualizer.show()

    '''DBSCAN'''   
    # neighbors = NearestNeighbors(n_neighbors=24)
    # neighbors_fit = neighbors.fit(df)
    # distances, indices = neighbors_fit.kneighbors(df)
    # # print(len(distances[0]))
    # distances = np.sort(distances, axis=0)
    # distances = distances[:, 1]
    # plt.plot(distances)
    # plt.title('Nearest distance among all the data points')
    # plt.xlabel('Data points')
    # plt.ylabel('distances')
    # plt.show()

def silhouette(df, label):
    '''silhoutte'''
    score = metrics.silhouette_score(df, label, metric='euclidean')
    print(f'Silhoutte Coefficient: {score}')
    
    # visualizer = SilhouetteVisualizer(model, colors='yellowbrick')
    
    # visualizer.fit(df)
    # visualizer.poof()
    

    
def k_means(df, genre):
    '''
    k means clustering
    https://www.statology.org/k-means-clustering-in-python/
    
    hiearchical clustering
    https://hands-on.cloud/implementation-of-hierarchical-clustering-using-python/
    
    '''

    start_time = time.time()
    '''k-means'''
    # model = KMeans(n_clusters=4, random_state=0)
    # model.fit(df)
    # df['cluster'] = pd.Series(model.labels_)
    # predicted_cluster = model.labels_
    '''Hiearchical Clustering'''
    # model = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='average')
    # predicted_cluster = model.fit_predict(df)
    '''DBSCAN'''
    # model = DBSCAN(eps = 1.8, min_samples = 24).fit(df)
    # predicted_cluster = model.labels_
    '''GMM'''
    model = GaussianMixture(n_components=3).fit(df)
    '''Birch'''
    # model = Birch(n_clusters=5).fit(df)
    predicted_cluster = model.predict(df)
    end_time = time.time()
    df['cluster'] = pd.DataFrame(predicted_cluster)
    df['genre'] = genre
    print(df['cluster'].value_counts())
    df.dropna(axis=0, inplace=True)
    #print(df['genre'].isnull().sum())
    # gb = df.groupby('cluster')
    
    # for c in df['cluster'].unique():
    #     print(c)
    #     lst = [(gen, count) for gen, count in gb.get_group(c)['genre'].value_counts().items()]
    #     print(lst)
    #     plot_bar([lst[i][0] for i in range(len(lst))], [lst[i][1] for i in range(len(lst))], c)
    
    df.drop(columns=['cluster', 'genre'], inplace=True)
    print(f'Total time spent for this model is: {(end_time - start_time) * 10 ** 3} ms')
    return model, predicted_cluster

def build_ans(df, genre):
    df['genre'] = genre

    df['genre'] = df['genre'].str.replace('Trap Metal', '1')
    df['genre'] = df['genre'].str.replace('Dark Trap', '0')
    df['genre'] = df['genre'].str.replace('Underground Rap', '0')# Underground Rap???, since Rap is reaplced as 0
    df['genre'] = df['genre'].str.replace('techhouse', '2')
    df['genre'] = df['genre'].str.replace('Rap', '0')
    df['genre'] = df['genre'].str.replace('Pop', '0')
    df['genre'] = df['genre'].str.replace('techno', '1')
    df['genre'] = df['genre'].str.replace('psytrance', '2')
    df['genre'] = df['genre'].str.replace('trance', '2')
    df['genre'] = df['genre'].str.replace('hardstyle', '2')
    df['genre'] = df['genre'].str.replace('trap', '0')
    df['genre'] = df['genre'].str.replace('dnb', '2')
    df['genre'] = df['genre'].str.replace('Emo', '2')
    df['genre'] = df['genre'].str.replace('Hiphop', '0')
    df['genre'] = df['genre'].str.replace('RnB', '0')

    df['genre'] = df['genre'].astype(int)

    ans = df['genre']
    df.drop(axis=0, columns='genre', inplace=True)
    print(df.isnull().sum())
    
    return ans

def evaluate(ans, predicted):
    '''
    given ans labels and predicted values, we can evaluate the result of clustering
    https://www.796t.com/content/1548088743.html
    
    '''
    
    print(f'Rand Index: {metrics.rand_score(ans, predicted)}')
    print(f'Normalized Mutual Information: {metrics.normalized_mutual_info_score(ans, predicted)}')
    print(f'Adjusted Mutual Information: {metrics.adjusted_mutual_info_score(ans, predicted)}')
    print(f'V Measure: {metrics.v_measure_score(ans, predicted)}')
    print(f'Fowlkes-Mallows Scores: {metrics.fowlkes_mallows_score(ans, predicted)}')
    
    matrix = metrics.confusion_matrix(ans, predicted)
    display = metrics.ConfusionMatrixDisplay(matrix)
    display.plot()
    plt.show()

def sampling(df):
    df_sample = df.sample(n=30000)
    df_sample.to_csv('sample.csv', index=False)
    return df_sample

def pca(df):
    # pca = PCA(n_components=13)
    # pca.fit(df)
    # variance = np.round(np.cumsum(pca.explained_variance_ratio_), decimals=3) * 100
    # print(variance)
    # plt.figure(figsize=(12, 6))
    # plt.ylabel('variance ratio')
    # plt.xlabel('# of features')
    # plt.title('PCA Analysis')
    # plt.plot(variance, marker='o')
    # plt.show()
    
    pca = PCA(n_components=6)
    pca_res = pca.fit_transform(df)
    df_pca = pd.DataFrame(pca_res, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6'])
    
    return df_pca



sample = False
if sample:
    read = pd.read_csv('sample.csv', low_memory=False)
else:
    read = pd.read_csv('genres_v2.csv', low_memory=False)
#read = sampling(read)
df, duration_min, genre = preprocess(read)
print(len(df))
# df['genre'] = genre
# print(gb.get_group('dnb'))

# df_pca = pca(df)

# elbow(df)

# df = df_pca
'''build clustering models, and return models and predicted cluster for each row of data'''
# model, predicted = k_means(df, genre)

# silhouette(df, predicted)

# '''build cluster ans labels'''
# ans = build_ans(df, genre)

# '''evaluation'''
# evaluate(ans, predicted)

# y = [2395.153522491455, 42682.422399520874, 23256.197929382324, 1520.604133605957, 38306.37454986572]
# x = ['k_means', 'Hierarchical Clustering', 'DBSCAN', 'GMM', 'BIRCH']
# plot_bar(x, y, 'Time Cost Comparison')

# y = [24352 / 42305.0, 26965 / 42305.0]
# x = ['k_means', 'PCA + k_means']
# plot_bar(x, y, 'Accuracy')