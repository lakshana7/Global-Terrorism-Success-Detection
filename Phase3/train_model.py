import numpy as np
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier as DTree_F

def neg_values(c):
    neg_val_col = []
    for i,j in df[c].value_counts().items():
        if i < 0:
            neg_val_col.append(i)
    return neg_val_col

# Handling the outliers
def handle_outliers(out_data_cols, df):
    for i in out_data_cols:
        Q1 = df[i].quantile(0.25)
        Q3 = df[i].quantile(0.75)
        IQR = Q3 - Q1
        df.loc[df[i] > Q3+(1.5*IQR), i] = np.mean(df[i])
        df.loc[df[i] < Q1-(1.5*IQR), i] = np.mean(df[i])

# Preprocessing the data
def preprocess_csv(org_df, f=False):
    new_df = org_df[['iyear', 'imonth', 'iday', 'country_txt', 'city', 'latitude', 'longitude', 'success', 'motive', 'attacktype1_txt', 'targtype1_txt', 'target1', 'weaptype1_txt', 'nkill', 'nkillus', 'nwound', 'nwoundus', 'ishostkid', 'nhostkid', 'ransom', 'ransomamt']]
    drop_col = ["motive", 'nkillus', 'nwoundus', 'ransom', 'ransomamt']

    # Dropping columns and dropping or filling NaNs
    df = new_df.drop(drop_col, axis=1)
    df.dropna(subset = ['latitude', 'longitude', 'target1', 'city'], inplace=True)
    df["nkill"].fillna(df["nkill"].mean(), inplace = True)
    df["nwound"].fillna(df["nwound"].mean(), inplace = True)
    cols = df.describe().columns
    for c, v in [('ishostkid', [-9.0]), ('nhostkid', [-99.0])]:
        df = df[df[c] != v[0]]

    # Dropping the 0 values in imonth and iday
    df = df[df['imonth'] != 0]
    df = df[df['iday'] != 0]
    df['HostKidsCount'] = df['ishostkid']
    df['HostKidsCount'] = df[df['HostKidsCount'] == 1.0]["nhostkid"]

    # Dropping the columns ishostkid and nhostkid
    df = df.drop(['ishostkid', 'nhostkid'], axis=1)

    # Replacing the NaNs with mode
    df['HostKidsCount'].fillna(df['HostKidsCount'].mode()[0], inplace = True)
    df.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country','city':'City',
                       'latitude':'Latitude','longitude':'Longitude','success':'Success','attacktype1_txt':'AttackType',
                       'targtype1_txt': 'TargetType','target1':'Target','weaptype1_txt':'WeaponType','nkill':'Killed',
                       'nwound':'Wounded','ishostkid':'IsHostKid','nhostkid':'No. Host Kid'},inplace=True)

    df.sort_values(by='Year', ascending=True, inplace =True)
    df.reset_index()
    df["AttackType"] = df["AttackType"].replace("Unknown", df["AttackType"].mode()[0])


    # Replacing the Unknown type in WeaponType and TargetType columns with modes of respective values
    df['Casuality'] = df['Wounded']+df['Killed']
    df["WeaponType"] = df["WeaponType"].replace("Unknown", df["WeaponType"].mode()[0])
    df["TargetType"] = df["TargetType"].replace("Unknown", df["TargetType"].mode()[0])
    df['Count'] = 1
    df["City"] = df["City"].replace("Unknown", df["City"].mode()[0])

    out_data_cols = ['Killed', 'Wounded', 'HostKidsCount']
    handle_outliers(out_data_cols, df)
    df = df.drop(columns='Count')   
    Numeric_col = df.select_dtypes(include = ['int', 'float']).columns

    # Normalizing the data
    for i in Numeric_col:
        min = df[i].min()
        max = df[i].max()
        df[i] = (df[i] - min)/(max - min)
    df = pd.get_dummies(df,columns = ['AttackType','WeaponType'])

    X = df.iloc[:,10:33]
    y = df['Success']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_test_scaled,y_test
 
def Algorithms_6(X_train, y_train, X_test, y_test):
    #Implementing Decision Tree Classifier
    A6 = DTree_F(criterion="log_loss", splitter="best", min_samples_split=3)
    A6.fit(X_train, y_train)
    decision_tree_predictions = A6.predict(X_test)
    with open('decision_tree.pkl', 'wb') as file:
        pickle.dump(A6, file)

print('Started training the model!')
org_df = pd.read_csv('globalterrorismdb_0718dist.csv', encoding='latin1')
X_train_scaled, y_train, X_test_scaled,y_test = preprocess_csv(org_df)
res = Algorithms_6(X_train_scaled, y_train, X_test_scaled,y_test)
print('Model has been trained!')