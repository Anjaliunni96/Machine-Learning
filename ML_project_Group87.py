import pandas as pd  
import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeRegressor
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
import time
start_time = time.time()

#loading the training dataset
df=pd.read_csv('training.csv')
 
df=df.drop(df.loc[:, 'm2379.76':'m2352.76'].columns, axis = 1) 

#one hot coding for soil type
soil= df.Depth.str.get_dummies()
df =pd.concat([df,soil],axis=1) 

#smoothening of dataset using savgol filter
spectra= [m for m in list(df.columns) if m[0]=='m'] 
smoothed_2dg = savgol_filter(df[spectra], window_length = 11, polyorder = 3, deriv=1)
df_smooth=pd.DataFrame(smoothed_2dg,columns=spectra)
df[spectra]=df_smooth

#derivative 
train_smooth_1stderiv=df_smooth
smoothed_1dg = savgol_filter(train_smooth_1stderiv[spectra], window_length = 11, polyorder = 3, deriv=1) #running poliamial smoother with deravatives 1
smooth_train1=pd.DataFrame(smoothed_1dg,columns=spectra)
df[spectra]=smooth_train1

#standard deviation 
threshold = 0.003
df.drop(df.std()[df.std() < threshold].index.values, axis=1)

#correlation matrix
c = df.corr().abs()

##################Calcium###################

# eliminating features having low correlation
ca=c[['Ca']]
ca=ca[ca['Ca'] >0.25].reset_index()
feat_ca=list(ca['index'].unique())
df_ca=df[[*feat_ca]]
df_ca=df_ca.drop([ 'pH', 'SOC', 'Sand','Ca'], axis = 1)

X=df_ca.copy()
y=df.Ca
col = list(X.columns)
clf = RandomForestRegressor (n_estimators=100, random_state=0, n_jobs=-1)
clf.fit(X,y)

sfm = SelectFromModel(clf, threshold=0.0001)
sfm.fit(X,y)
list1 = []
for feature_list_index in sfm.get_support(indices=True):
    list1.append(col[feature_list_index])

X=X[[*list1]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
df_ca_pa=df_ca.copy()
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#PCA
pca = PCA(n_components=len(list1))
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

base_model = RandomForestRegressor(n_estimators = 100, random_state = 42)
base_model.fit(X_train, y_train)
y_predicted = base_model.predict(X_test)
y_test = y_test.to_numpy()
calcium=np.sqrt(mean_squared_error(y_test, y_predicted))
################################Phosphorus P###########################
p = c.copy()
pa=p[['P']]
pa=pa[pa['P'] > 0.1].reset_index()
feat_pa=list(pa['index'].unique())
df_pa=df[[*feat_pa]]
df_pa=df_pa.drop(['P'], axis = 1)
X=df_pa.copy()
y=df.P
col = list(X.columns)
clf = RandomForestRegressor (n_estimators=100, random_state=0, n_jobs=-1)

clf.fit(X,y)
sfm = SelectFromModel(clf, threshold=0.0001)

sfm.fit(X,y)
list1 = []
for feature_list_index in sfm.get_support(indices=True):
    list1.append(col[feature_list_index])

X1=X[[*list1]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#PCA
pca = PCA(n_components=13)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

#model
base_model =  RandomForestRegressor (n_estimators=100, random_state=0, n_jobs=-1)
base_model.fit(X_train, y_train)
y_predicted = base_model.predict(X_test)
y_test = y_test.to_numpy()
phosphorous=np.sqrt(mean_squared_error(y_test, y_predicted))
################################ SAND ##############################

s = c.copy()
sa=s[['Sand']]
sa=sa[sa['Sand'] >0.10].reset_index()
feat_sa=list(sa['index'].unique())
df_sa=df[[*feat_sa]]
df_sa=df_sa.drop(['SOC', 'Sand','Ca'], axis = 1)
X=df_sa.copy()
y=df.Sand
col = list(X.columns)
clf = RandomForestRegressor (n_estimators=100, random_state=0, n_jobs=-1)
clf.fit(X,y)

sfm = SelectFromModel(clf, threshold=0.00025)

sfm.fit(X,y)
list1 = []
for feature_list_index in sfm.get_support(indices=True):
    list1.append(col[feature_list_index])

X=X[[*list1]]
#RFE automatic number of features selected
rfecv1 = RFECV(estimator=DecisionTreeRegressor(), step=1, cv=5)
rfecv1 = rfecv1.fit(X, y)
rfecv1_rank_list = list(rfecv1.ranking_)
indices = [i for i, x in enumerate(rfecv1_rank_list) if x == 1]
list2 = [list1[i] for i in indices]
X=X[[*list2]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#model
base_model = RandomForestRegressor(n_estimators = 100, random_state = 42)
base_model.fit(X_train, y_train)
y_predicted = base_model.predict(X_test)
y_test = y_test.to_numpy()

sand=np.sqrt(mean_squared_error(y_test, y_predicted))
############################## pH ###################################

ph=c[['pH']]
ph=ph[ph['pH'] >0.25].reset_index()
feat_ph=list(ph['index'].unique())
df_ph=df[[*feat_ph]]

df_ph=df_ph.drop([ 'pH','Ca'], axis = 1)

X_ph=df_ph.copy()
y_ph=df.pH
col_ph = list(X_ph.columns)
clf_ph = RandomForestRegressor (n_estimators=150, random_state=0, n_jobs=-1)
clf_ph.fit(X_ph,y_ph)

sfm_ph = SelectFromModel(clf_ph, threshold=0.001)

sfm_ph.fit(X_ph,y_ph)
list1_ph = []
for feature_list_index_ph in sfm_ph.get_support(indices=True):
    list1_ph.append(col_ph[feature_list_index_ph])

X_ph=X_ph[[*list1_ph]]

X_train_ph, X_test_ph, y_train_ph, y_test_ph = train_test_split(X_ph, y_ph, test_size=0.2, random_state=0)

df_pf_pa=df_ph.copy()
sc_ph = StandardScaler()
X_train_ph = sc_ph.fit_transform(X_train_ph)
X_test_ph = sc_ph.transform(X_test_ph)

pca_ph = PCA(n_components=len(list1_ph))
X_train_ph = pca_ph.fit_transform(X_train_ph)
X_test_ph = pca_ph.transform(X_test_ph)

explained_variance_ph = pca_ph.explained_variance_ratio_

base_model_ph = RandomForestRegressor(n_estimators = 10, random_state = 42)
base_model_ph.fit(X_train_ph, y_train_ph)
y_predicted_ph = base_model_ph.predict(X_test_ph)
ph_mse=np.sqrt(mean_squared_error(y_test_ph, y_predicted_ph))
################### SOC ###############

so = c.copy()
soc=so[['SOC']]
soc=soc[soc['SOC'] >0.1].reset_index()
feat_soc=list(soc['index'].unique())
df_soc=df[[*feat_soc]]

df_soc=df_soc.drop([ 'Sand','SOC','P','Ca'], axis = 1)

X=df_soc.copy()
y=df.SOC
col = list(X.columns)
clf = RandomForestRegressor (n_estimators=100, random_state=0, n_jobs=-1)
clf.fit(X,y)
sfm = SelectFromModel(clf, threshold=0.00025)

sfm.fit(X,y)
list1 = []
for feature_list_index in sfm.get_support(indices=True):
    list1.append(col[feature_list_index])
X1=X[[*list1]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#PCA
pca = PCA(n_components=40)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

#model
base_model = RandomForestRegressor(n_estimators = 250, random_state = 42)
base_model.fit(X_train, y_train)
y_predicted = base_model.predict(X_test)
y_test = y_test.to_numpy()
soc_mse=np.sqrt(mean_squared_error(y_test, y_predicted))

sum_of_errors=calcium+phosphorous+ph_mse+sand+soc_mse
MCRMSE=sum_of_errors/5

print()
print('Mean Columnwise Root Mean Square Erro (MCRMSE) : ', MCRMSE)
print()
print("---Total runtime %s seconds ---" % (time.time() - start_time))
