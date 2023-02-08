import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt 
import time 
from pprint import pprint
from scipy import stats
from sklearn import preprocessing, svm
from sklearn.svm import SVC 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import ClusterMixin
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, Birch
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from yellowbrick.cluster import SilhouetteVisualizer, KElbowVisualizer
from yellowbrick.cluster import KElbowVisualizer
import seaborn as sns
import scipy.cluster.hierarchy as sch

from imblearn.pipeline import make_pipeline
from imblearn import datasets
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, GridSearchCV, train_test_split, KFold, StratifiedKFold
from sklearn import metrics
from sklearn.metrics import recall_score, roc_auc_score, f1_score
from sklearn.inspection import permutation_importance
import gc

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
    '''transform duration_ms into duration_min'''
    df = df.drop(axis = 1, columns = ['id', 'type', 'uri', 'track_href', 'analysis_url', 'song_name', 'Unnamed: 0', 'title'])
    df['duration_ms'] = df['duration_ms'] / 60000.0
    df = df.rename({'duration_ms':'duration_min'}, axis=1)


    '''transform nominal variable - genre into number'''
    label_encoder = LabelEncoder()
    df['genre'] = label_encoder.fit_transform(df['genre'])
    # print(label_encoder.classes_)#print the original name of data before transformation
    
    '''remove outliers'''
    for col in df.columns:
        if col != 'genre': df = find_outlier(df, col)
    df.dropna(inplace=True)
   
    '''Normalize features''' 
    df[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 'duration_min']] = StandardScaler().fit_transform(df[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 'duration_min']])
    
    return df, label_encoder.classes_

def find_outlier(df, col):
    z = np.abs(stats.zscore(df[col]))
    index_lst = np.where(z > 3)
    # print(index_lst[0][0])
    for idx in index_lst[0]: df.at[idx, col] = np.nan
    
    return df

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
    # model = KMeans()
    # model = AgglomerativeClustering()
    '''Birch'''
    model = Birch(n_clusters=None)
    
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
    

    
def k_means(df, genre, genre_name, option):
    '''
    k means clustering
    https://www.statology.org/k-means-clustering-in-python/
    
    hiearchical clustering
    https://hands-on.cloud/implementation-of-hierarchical-clustering-using-python/
    
    '''

    # start_time = time.time()
    if option == 'k-means':
        '''k-means'''
        model = KMeans(n_clusters=4, random_state=0)
        model.fit(df)
        predicted_cluster = model.labels_
    elif option == 'Hiearchical':
        '''Hiearchical Clustering'''
        model = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='average')
        predicted_cluster = model.fit_predict(df)
    elif option == 'DBSCAN':
        '''DBSCAN'''
        model = DBSCAN(eps = 0.7, min_samples = 12).fit(df)#eps determined by k nearest neighbor; min_samples = 2* number of features
        predicted_cluster = model.labels_ 
    elif option == 'Birch':
        '''Birch'''
        model = Birch(n_clusters=7).fit(df)
        predicted_cluster = model.predict(df)     
    elif option == 'GMM':
        '''GMM'''
        model = GaussianMixture(n_components=4).fit(df)
        predicted_cluster = model.predict(df)              
    
    # end_time = time.time()
    
    df['cluster'] = pd.DataFrame(predicted_cluster)
    df['genre'] = genre
    print(df['cluster'].value_counts())
    df.dropna(axis=0, inplace=True)#some data is not classified into a cluster
    # print(df.isnull().sum())
    gb = df.groupby('cluster')
    
    for c in df['cluster'].unique():
        print(c)
        lst = [(gen, count) for gen, count in gb.get_group(c)['genre'].value_counts().items()]
        print(lst)
        plot_bar([genre_name[lst[i][0]] for i in range(len(lst))], [lst[i][1] for i in range(len(lst))], c)
    
    predicted_cluster = df['cluster']
    # df.drop(columns=['cluster', 'genre'], inplace=True)
    # print(f'Total time spent for this model is: {(end_time - start_time) * 10 ** 3} ms')
    # return model, predicted_cluster
    return model, df, predicted_cluster

# def build_ans(df, genre):
#     df['genre'] = genre

#     df['genre'] = df['genre'].str.replace('Trap Metal', '1')
#     df['genre'] = df['genre'].str.replace('Dark Trap', '0')
#     df['genre'] = df['genre'].str.replace('Underground Rap', '0')# Underground Rap???, since Rap is reaplced as 0
#     df['genre'] = df['genre'].str.replace('techhouse', '2')
#     df['genre'] = df['genre'].str.replace('Rap', '0')
#     df['genre'] = df['genre'].str.replace('Pop', '0')
#     df['genre'] = df['genre'].str.replace('techno', '1')
#     df['genre'] = df['genre'].str.replace('psytrance', '2')
#     df['genre'] = df['genre'].str.replace('trance', '2')
#     df['genre'] = df['genre'].str.replace('hardstyle', '2')
#     df['genre'] = df['genre'].str.replace('trap', '0')
#     df['genre'] = df['genre'].str.replace('dnb', '2')
#     df['genre'] = df['genre'].str.replace('Emo', '2')
#     df['genre'] = df['genre'].str.replace('Hiphop', '0')
#     df['genre'] = df['genre'].str.replace('RnB', '0')

#     df['genre'] = df['genre'].astype(int)

#     ans = df['genre']
#     df.drop(axis=0, columns='genre', inplace=True)
#     print(df.isnull().sum())
    
#     return ans

def build_answer(df, genre_name, cluster_num):
    cluster_idx = dict()#genre cluster pair
    
    for key in df.groupby('genre').groups.keys(): cluster_idx[genre_name[key]] = np.array([0] * cluster_num)
    for group_name, group_df in df.groupby('cluster'): 
        genre_lst = genre_name[group_df['genre']] #genre list for this cluster
        for genre in genre_lst: cluster_idx[genre][int(group_name)] += 1 #calculate the number of appearance of this genre in this cluster
    # print(cluster_idx)
    
    for genre, count_lst in cluster_idx.items():#find the genre's max value among the clusters to be ground truth
        cluster_idx[genre] = np.argmax(cluster_idx[genre])
    # print(cluster_idx)
    
    df['ground_truth'] = df['genre'].apply(lambda x: cluster_idx[genre_name[x]]) #build the corresponding cluster for each data
    # print(df['ground_truth'])
    
    return df['ground_truth']
    

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
    '''
    Stratified samping
    sampling according to the propotion of data of genre
    '''
    sampling_num = dict()
    sample_df = pd.DataFrame()
   
    total_entry = float(len(df))
    count_lst = [int(20000 * (count / total_entry)) for count in df['genre'].value_counts().tolist()]
    name_lst = df['genre'].value_counts().index.tolist()
    for genre_name, genre_count in zip(name_lst, count_lst): sampling_num[genre_name] = genre_count
    
    for group, gb_df in df.groupby('genre'):
        sample_df = pd.concat([sample_df, gb_df.sample(n=sampling_num[group])])
    
    sample_df.to_csv('sample.csv', index=False)
    return sample_df

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

def f_importances(model, X_train, y_train, X_test, y_test):
    #Feature importance
    '''for random forest'''
    features = model.feature_importances_ #for Random Forest
    print(f"Features:\n{features}")
    '''for rbf SVM'''
    perm_importance = permutation_importance(model, X_test, y_test)
    sorted_idx = perm_importance.importances_mean.argsort()
    features = np.array(X_train.columns)
    plt.barh(features[sorted_idx], perm_importance.importances_mean[sorted_idx])
    plt.xlabel("Permutation Importance")
    plt.tight_layout()

def f_importances_linear(coef, names):
    '''Plot feature importance for linear SVM'''
    imp = np.abs(np.array(coef))
    imp,names = zip(*sorted(zip(imp,names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.tight_layout()
    plt.show()
    

def data_spliting(X, y):
    '''
    input features(X) and labels(y)
    do k-fold validation and oversampling on each fold of training data
    '''
    accuracy_arr = np.zeros(5, dtype=float)
    kf = StratifiedKFold(n_splits = 5)
    for fold_num, (train_index, test_index) in enumerate(kf.split(X, y)):#kf.split returns two arrays of train indices and test indices
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        X_test =  X.iloc[test_index]
        y_test = y.iloc[test_index]
        #Smote
        smote = SMOTE()
        X_train_oversampled, y_train_oversampled = smote.fit_resample(X_train, y_train)
        print(f'Fold {fold_num}')
        #Model building and Evaluation
        accuracy = build_class_model(X_train_oversampled, y_train_oversampled, X_test, y_test)        
        
        # build_class_model(X_train, y_train, X_test, y_test)  
        accuracy_arr[fold_num] = accuracy
        print(f"fold: {fold_num}\tAccuracy: {accuracy_arr[fold_num]}")
    print(f"Overall accuracy Score:\t{np.average(accuracy_arr)}")

def random_tuning(train_features, train_labels):
    '''narraow down parameter sets for us to do grid search'''
    '''create parameter grid'''
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
    pprint(random_grid)
    
    '''instatiate random search and fit it# Use the random grid to search for best hyperparameters'''
    # First create the base model to tune
    rf = RandomForestClassifier()
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = None)
    # Fit the random search model
    rf_random.fit(train_features, train_labels)
    gc.collect()
    print(rf_random.best_params_)

def build_class_model(X_train, y_train, X_test, y_test):
    '''Random Forest'''
    # params = {'n_estimators': 1600, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 15, 'bootstrap': True}
    # model = RandomForestClassifier(**params)
    '''SVM'''
    # params = {'C': 1000, 'decision_function_shape': 'ovo', 'gamma': 0.01, 'kernel': 'rbf'}
    # params = {'C': 10, 'decision_function_shape': 'ovo', 'gamma': 1, 'kernel': 'linear'}
    # model = SVC(**params)
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)
    
    '''AdaBoost'''
    params = {'C': 10, 'decision_function_shape': 'ovo', 'gamma': 1, 'kernel': 'linear', 'probability': True}
    model = AdaBoostClassifier(svm.SVC(**params),n_estimators=50, learning_rate=1.0, algorithm='SAMME')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    '''Feature importance'''
    # f_importances(model, X_train, y_train, X_test, y_test)
    # f_importances_linear(model.coef_[0], X_train.columns)
    
    
    #evaluation
    # print(metrics.confusion_matrix(y_test, y_pred))
    # print('Performance:')
    # matrix = metrics.confusion_matrix(y_test, y_pred)
    # display = metrics.ConfusionMatrixDisplay(matrix)
    # display.plot()
    # plt.show()

    # print(f'Accuracy: {model.score(X_test, y_test)}')
    # print(f'Recall: {recall_score(y_test, y_pred, average = None)}')
    # print(f'F meausre: {f1_score(y_pred, y_test, average=None)}')
    print(metrics.classification_report(y_test, y_pred))
    return model.score(X_test, y_test) #return accuracy score of the model

def gridSearch_rf(X_train, y_train, X_test, y_test):
    '''Grid search for Random Forest'''
    rf = RandomForestClassifier(random_state = 42)
    params = {'n_estimators': [1500, 1600, 1700], 'min_samples_split': [10], 'min_samples_leaf': [1], 'max_features': ['sqrt'], 'max_depth': [15, 20, 25], 'bootstrap': [True]}
    grid = GridSearchCV(rf, param_grid=params, cv=5, 
                          scoring='roc_auc_ovr').fit(X_train, y_train)
    print(f"Best parameter sets: {grid.best_params_}")
    best_rf = (grid.best_estimator_).fit(X_train, y_train)#using training data to train model of better parameters
    
    y_pred = best_rf.predict(X_test)
    print("Performance of the optimal parameters on Random Forest")
    print(metrics.classification_report(y_test, y_pred))
    
def gridSearch_svm(X_train, y_train, X_test, y_test):
    '''Grid search for multi-class SVM'''
    param_grid = {'C': [0.1, 1, 10, 100, 1000], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['linear'],#could be rbf
              'decision_function_shape':['ovo']} 
  
    grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
    
    # fitting the model for grid search
    grid.fit(X_train, y_train)
    
    # print best parameter after tuning
    print(grid.best_params_)#overall parameters
    
    # print how our model looks after hyper-parameter tuning
    print(grid.best_estimator_)
    
    grid_predictions = grid.predict(X_test)
    
    # print classification report
    print(metrics.classification_report(y_test, grid_predictions))



sample = False #true when hiearchical clustering
if sample:
    read = pd.read_csv('sample.csv', low_memory=False)
else:
    read = pd.read_csv('genres_v2.csv', low_memory=False)
# read = sampling(read)
df, genre_name = preprocess(read)
# X = df[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 'duration_min']]
# y = df['genre']
genre = df['genre']
df = df[['danceability', 'loudness', 'speechiness', 'instrumentalness', 'valence', 'duration_min']]


# df_pca = pca(df)
gc.collect()
# elbow(df)

# df = df_pca
'''build clustering models, and return models and predicted cluster for each row of data'''
model, df_clustered, predicted = k_means(df, genre, genre_name, option = 'GMM')

silhouette(df_clustered, predicted)

# '''build cluster ans labels'''
ans = build_answer(df_clustered, genre_name, cluster_num = 7)
# ans = build_ans(df, genre)

# '''evaluation'''
evaluate(ans, predicted)

# y = [2395.153522491455, 42682.422399520874, 23256.197929382324, 1520.604133605957, 38306.37454986572]
# x = ['k_means', 'Hierarchical Clustering', 'DBSCAN', 'GMM', 'BIRCH']
# plot_bar(x, y, 'Time Cost Comparison')

# y = [24352 / 42305.0, 26965 / 42305.0]
# x = ['k_means', 'PCA + k_means']
# plot_bar(x, y, 'Accuracy')


'''Classification'''

# X = df[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 'duration_min']]
# y = df['genre']

'''train-test split'''
# X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=104,  test_size=0.25, shuffle=True)
'''Find better hyperparameters'''
# random_tuning(X_train, y_train)
# gridSearch_svm(X_train, y_train, X_test, y_test)
# build_class_model(X_train, y_train, X_test, y_test)

'''k-fold + SMOTE + Model building'''
# data_spliting(X, y)

'''boosting'''