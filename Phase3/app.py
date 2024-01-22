# Importing the necessary libraries
import base64
import graphviz
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import subprocess

from flask import Flask,render_template, request, jsonify
from io import BytesIO
from io import StringIO
from sklearn.preprocessing import StandardScaler
from sklearn.tree import export_graphviz

attack_types = [['armed assault', 'AttackType_Armed Assault'], ['assassination', 'AttackType_Assassination'], ['bombing/explosion', 'AttackType_Bombing/Explosion'], ['facility/infrastructure attack', 'AttackType_Facility/Infrastructure Attack'],
['hijacking', 'AttackType_Hijacking'], ['hostage taking barricade', 'AttackType_Hostage Taking (Barricade Incident)'], ['hostage taking kidnapping', 'AttackType_Hostage Taking (Kidnapping)'], ['unarmed assault', 'AttackType_Unarmed Assault']]

weapon_types = [['biological', 'WeaponType_Biological'], ['chemical', 'WeaponType_Chemical'], ['explosives', 'WeaponType_Explosives'], ['fake weapons', 'WeaponType_Fake Weapons'],
['firearms', 'WeaponType_Firearms'], ['incendiary', 'WeaponType_Incendiary'], ['melee', 'WeaponType_Melee'], ['other', 'WeaponType_Other'],
['radiological', 'WeaponType_Radiological'], ['sabotage equipment', 'WeaponType_Sabotage Equipment'], ['vehicle', 'WeaponType_Vehicle (not to include vehicle-borne explosives, i.e., car or truck bombs)']]

# Finding out if there is a negative number of hostages
def neg_values(c):
    neg_val_col = []
    for i,j in df[c].value_counts().items():
        if i < 0:
            neg_val_col.append(i)
    return neg_val_col

# Get pieplot for Attack types
def get_pieplot(df):
    category_counts = df['AttackType'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 4))
    # Retrieve most used attack type
    most_used_attack = category_counts.idxmax()

    ax.pie(category_counts, labels = category_counts.index, autopct='%1.1f%%', radius = 1)
    ax.set_title('Percentage of Attack Types')

    # Saving the pie plot with BytesIO object and encoding the image to base64
    image = BytesIO()
    fig.savefig(image, format='png')
    image.seek(0)
    img_bytes = image.read()
    image_path = f'data:image/png;base64,{base64.b64encode(img_bytes).decode()}'
    
    return image_path, most_used_attack

# Preprocess the dataset uploaded by the user from UI
def preprocess_csv(org_df, f=False):
    new_df = org_df[['iyear', 'imonth', 'iday', 'country_txt', 'city', 'latitude', 'longitude', 'success', 'motive', 'attacktype1_txt', 'targtype1_txt', 'target1', 'weaptype1_txt', 'nkill', 'nkillus', 'nwound', 'nwoundus', 'ishostkid', 'nhostkid', 'ransom', 'ransomamt']]
    drop_col = ["motive", 'nkillus', 'nwoundus', 'ransom', 'ransomamt']

    # Dropping columns and filling NaNs
    df = new_df.drop(drop_col, axis=1)
    df["nkill"].fillna(df["nkill"].mean(), inplace = True)
    df["nwound"].fillna(df["nwound"].mean(), inplace = True)

    cols = df.describe().columns
    for c, v in [('ishostkid', [-9.0]), ('nhostkid', [-99.0])]:
        df = df[df[c] != v[0]]

    # Creating column for Host Kids Count
    df['HostKidsCount'] = df['ishostkid']
    df['HostKidsCount'] = df[df['HostKidsCount'] == 1.0]["nhostkid"]

    # Dropping the redundant columns ishostkid and nhostkid
    df = df.drop(['ishostkid', 'nhostkid'], axis=1)

    # Replacing the NaNs with mode
    df['HostKidsCount'].fillna(df['HostKidsCount'].mode()[0], inplace = True)
    # Renaming the columns
    df.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country','city':'City',
                       'latitude':'Latitude','longitude':'Longitude','success':'Success','attacktype1_txt':'AttackType',
                       'targtype1_txt': 'TargetType','target1':'Target','weaptype1_txt':'WeaponType','nkill':'Killed',
                       'nwound':'Wounded','ishostkid':'IsHostKid','nhostkid':'No. Host Kid'},inplace=True)

    df.sort_values(by='Year', ascending=True, inplace =True)
    df.reset_index()
    df["AttackType"] = df["AttackType"].replace("Unknown", df["AttackType"].mode()[0])

    # Replacing the Unknown type in WeaponType and TargetType columns with modes of respective values
    df['Casuality'] = df['Wounded'] + df['Killed']
    df["WeaponType"] = df["WeaponType"].replace("Unknown", df["WeaponType"].mode()[0])
    df["TargetType"] = df["TargetType"].replace("Unknown", df["TargetType"].mode()[0])
    df['Count'] = 1
    df["City"] = df["City"].replace("Unknown", df["City"].mode()[0])

    df = df.drop(columns='Count')
    # Normalizing the data
    Numeric_col = df.select_dtypes(include = ['int', 'float']).columns
    for i in Numeric_col:
        min = df[i].min()
        max = df[i].max()
        df[i] = (df[i] - min)/(max - min)
    # Get pie plot for attack type
    attack_pie_img, most_used_attack = get_pieplot(df)
    df = pd.get_dummies(df,columns = ['AttackType','WeaponType'])

    # Getting the required data columns and scaling
    X = df.iloc[:,10:33]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X)

    return X_test_scaled, attack_pie_img, most_used_attack

# Get the bar plot for predictions
def get_barplot(predictions):
    # Drawing the bar plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(['Failure', 'Success'], np.bincount(predictions.astype(int)), color=['green', 'red'])
    ax.set_xlabel('Attack')
    ax.set_ylabel('Instances')
    ax.set_title('Count of Failure and Success Attacks')

    # Saving the bar plot with a BytesIO object and encoding it to base64
    image = BytesIO()
    fig.savefig(image, format='png')
    image.seek(0)
    img_bytes = image.read()
    image_path = f'data:image/png;base64,{base64.b64encode(img_bytes).decode()}'

    return image_path

# Function for returning suggestions based on the most_used_attack
def get_sug(most_used_attack):
    print("Most used attack:", most_used_attack)
    if most_used_attack in 'AttackType_Armed Assault':
        sug = 'Deploy more security forces, metal detectors to reduce the ' + most_used_attack + ' which is most occuring in the given csv'
    elif most_used_attack in 'AttackType_Assassination':
        sug = 'Assassination attack type might be more like a targeted attack of a celebrity, keep an eye on mass gatherings and lax security measures might be a concern'
    elif most_used_attack in 'AttackType_Bombing/Explosion':
        sug = 'Use metal detectors and scanners to detect radio active substances in crowded places like airports, stadiums, railways stations etc..'
    elif most_used_attack in 'AttackType_Facility/Infrastructure Attack':
        sug = 'Security flaws are a reason, bolstering the infrastruce security by deploying security cameras, identity management can solve the issue.'
    elif most_used_attack in 'AttackType_Hijacking':
        sug = 'Assassination attack type might be more like a targeted attack of a celebrity, keep an eyen on mass gatherings and lax security measures might be a concern'
    elif most_used_attack in 'AttackType_Hostage Taking':
        sug = 'Kidnapping and Barricade incident are the major reasons, always keep an eye on criminals with bad crime history'
    else:   # AttackType_Unarmed Assault
        sug = 'Deploying security guards, cops and maintaining the required levels of security measures will put ' + most_used_attack + ' under check'

    return sug

app=Flask(__name__, static_folder='static')

@app.route('/')
def home():
    # Render home.html, user lands on this page
    return render_template("home.html", image_path='loading.gif')

@app.route('/train_model', methods=['GET', 'POST'])
def train_model():
    # This route gets hit when user clicks 'Train Model' button embedded in home.html
    
    # Calls train_model.py script to train the model for the original dataset.
    subprocess.run(['python', 'train_model.py'])
    
    return jsonify(message="Model is trained!")

@app.route('/upload_csv', methods=['POST'])
def upload():
    # This route gets hit when user uploads a test_dataset and clicks 'Upload' to explore the predicitons of our model
    
    # Retrieve the saved model from pickle, which was done in train_model.py
    with open('decision_tree.pkl', 'rb') as file:
        decision_tree = pickle.load(file)
     
    # If user clicks Upload with inserting a file, it warns the user, with 'No file found!'
    if 'csv_file' not in request.files:
        return render_template('result.html', message='No file found!')

    # Get the uploaded csv file
    csv_file = request.files['csv_file']

    # If empty filename is shown, warns the user accordingly
    if csv_file.filename == '':
        return render_template('result.html', message='No selected file!')

    # File is correct, process the data, make predictions and render the results in result.html
    if csv_file:
        # Reads the csv file as dataframe using pandas
        df = pd.read_csv(csv_file, encoding='latin1')
        
        # Make required preprocessing, if required data is not found drops and performs necessary scaling and returns
        # X data for input, pie plot for attack types, most used attack
        X, attack_pie_img, most_used_attack = preprocess_csv(df, f=True)
        predictions = decision_tree.predict(X)

        # Converting the predictions to a list for rendering in HTML
        prediction_list = predictions.tolist()
        
        # Count number of success and failure cases
        s_c = 0
        for i in prediction_list:
            if i== 1.0:
                s_c += 1
        f_c = len(prediction_list)-s_c
        
        # Get Bar plot for predictions
        bar_img = get_barplot(predictions)
        
        # Get suggestions based on the most_used_attack
        sug = get_sug(most_used_attack)
        
        # Creating Decision Tree for viewing
        dot_data = export_graphviz(decision_tree, out_file=None,
                               feature_names=[f"Feature {i}" for i in range(X.shape[1])],
                               class_names=[str(i) for i in decision_tree.classes_],
                               filled=True, rounded=True, special_characters=True)
        print(dot_data)
        graph = graphviz.Source(dot_data)
        graph.render("DecisionTree")  # Saves the visualization to a file
        graph.view("DecisionTreeView")
        
        return render_template('result.html', csv=True, bar_img=bar_img, pie_img=attack_pie_img, sug=sug, predictions=prediction_list, fail_count=f_c, success_count=s_c)

columns = ['Killed', 'Wounded', 'HostKidsCount', 'Cauality', 'AttackType_Armed Assault', 'AttackType_Assassination', 
'AttackType_Bombing/Explosion', 'AttackType_Facility/Infrastructure Attack', 'AttackType_Hijacking', 'AttackType_Hostage Taking (Barricade Incident)', 'AttackType_Hostage Taking (Kidnapping)', 'AttackType_Unarmed Assault',
 'WeaponType_Biological', 'WeaponType_Chemical', 'WeaponType_Explosives', 'WeaponType_Fake Weapon', 'WeaponType_Firearms', 'WeaponType_Incendiary', 'WeaponType_Melee', 'WeaponType_Other', 'WeaponType_Radiological', 'WeaponType_Sabotage Equipment', 'WeaponType_Vehicle (not to include vehicle-borne explosives, i.e., car or truck bombs)']

@app.route('/submit', methods=['POST'])
def submit():
    # This route gets hit when user enters data sample and clicks 'Submit' to explore the predicitons of our model
    
    # Retrieve the saved model from pickle, which was done in train_model.py
    with open('decision_tree.pkl', 'rb') as file:
        decision_tree = pickle.load(file)
        
    if request.method == 'POST':
        # Retrieve the input data
        num_killed = request.form['num_killed']
        num_wounded = request.form['num_wounded']
        num_hostkids = request.form['num_hostkids']
        num_casuality = int(num_killed) + int(num_wounded)
        attacktype = request.form['attacktype']
        weapontype = request.form['weapon_type']

        data = [[num_killed, num_wounded, num_hostkids, num_casuality]]
        # Setting the corresponding attack_types and weapon_types to 1 if it is used, if not 0
        print(str(attacktype).lower(), str(weapontype).lower())
        for i in attack_types:
            print(i)
            if str(attacktype).lower() == i[0]:
                data[0].append(1)
                attack = i[1]
            else:
                data[0].append(0)
        weapon = ''
        for i in weapon_types:
            print(i)
            if str(weapontype).lower() == i[0]:
                data[0].append(1)
                weapon = i[1]
            else:
                data[0].append(0)
        
        # Create dataframe based on input data
        df = pd.DataFrame(data, columns=columns)
        X = df.iloc[:,:]
        print("X is:", df.iloc[0].tolist())
        
        # Scaling the data and make predictions
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X)
        X_test_scaled = scaler.transform(X)
        predictions = decision_tree.predict(X_test_scaled)
        
        # Creating Pie plot for attack type
        category_counts = df[weapon].value_counts()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.pie(category_counts, labels = [weapon], autopct='%1.1f%%', radius = 1.2)
        ax.set_title('Used Weapon Type')

        # Saving the plot with BytesIO object abd encoding to base64 for HTML embedding
        image = BytesIO()
        fig.savefig(image, format='png')
        image.seek(0)
        img_bytes = image.read()
        image_path = f'data:image/png;base64,{base64.b64encode(img_bytes).decode()}'
    
        # Get bar plot
        bar_img = get_barplot(predictions)

        # Converting predictions to a list for rendering in HTML
        prediction_list = predictions.tolist()
        
        # Count number of success and failure cases
        s_c = 0
        for i in prediction_list:
            if i== 1.0:
                s_c += 1
        f_c = len(prediction_list)-s_c
        
        # Get bar plots and suggestions
        bar_img = get_barplot(predictions)
        sug = get_sug(attack)
        
        # Creating Decision Tree for viewing
        dot_data = export_graphviz(decision_tree, out_file=None,
                               feature_names=[f"Feature {i}" for i in range(X_test_scaled.shape[1])],
                               class_names=[str(i) for i in decision_tree.classes_],
                               filled=True, rounded=True, special_characters=True)
        print(dot_data)
        graph = graphviz.Source(dot_data)
        graph.render("DecisionTree")  # Saves the visualization to a file
        graph.view("DecisionTreeView")  
        
        return render_template('result.html', datapoint=True, sug=sug, bar_img=bar_img, pie_img=image_path, attacktype=attacktype, fail_count=f_c, success_count=s_c, num_killed=num_killed, num_wounded=num_wounded)

if __name__=="__main__":
    app.run(debug=True)
