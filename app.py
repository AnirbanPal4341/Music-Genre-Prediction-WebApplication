# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 11:09:40 2021
@author: Anirban Pal
@project name: Music Genre Detection System

"""
import os
from flask import Flask,redirect,url_for,render_template,request
import pandas as pd
import numpy as np
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
import tensorflow.keras as keras
from keras.models import load_model
import csv 
import shutil

app = Flask(__name__)

app_root=os.path.dirname(os.path.abspath(__file__))
target=os.path.join(app_root,'static')
bin_folder=os.path.join(target,'bin')
saved_file="audio.wav"
saved_csv="UploadedMusic_Features_dataset.csv"
saved_img1=""
saved_img2=""

@app.route("/",methods=["POST","GET"])
def home():  
    shutil.rmtree(bin_folder,ignore_errors = True)
    os.mkdir(bin_folder)
    if request.method == "POST":
        file = request.files['soundFile']
        file_name=file.filename
        destination=f'./static/bin/{file_name}'
        file.save(destination)
        return redirect(url_for("analysis",audioName=file_name))
    else:
        return render_template("index.html")
    

@app.route("/about")
def about():
        return render_template("about.html")
    
@app.route("/audio-analysis/<audioName>",methods=["POST","GET"])
def analysis(audioName):
    print(audioName)
    ''' creating csv file for our input data and storing the extracted features '''
    audio_data=f'./static/bin/{audioName}'
    file = open('./static/bin/UploadedMusic_Features_dataset.csv', 'w', newline='')
    category=['filename', 'chroma_stft', 'rmse', 'spectral_centroid', 'spectral_bandwidth',
              'spectral_rolloff', 'zero_crossing_rate', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4',
              'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12',
              'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20']
    with file:
        writer = csv.writer(file)
        writer.writerow(category)
    
    x, sr = librosa.load(audio_data, mono=True , duration=30)
    chroma_stft = librosa.feature.chroma_stft(y=x, sr=sr)
    rmse = librosa.feature.rmse(y=x)
    spectral_centroid = librosa.feature.spectral_centroid(y=x, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=x, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=x, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(x)
    mfcc = librosa.feature.mfcc(y=x, sr=sr)
    feature_data = f'{audioName} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spectral_centroid)} {np.mean(spectral_bandwidth)} {np.mean(spectral_rolloff)} {np.mean(zero_crossing_rate)}'    
    for m in mfcc:
        feature_data += f' {np.mean(m)}'
    file = open('./static/bin/UploadedMusic_Features_dataset.csv', 'a', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(feature_data.split())
         
    data = pd.read_csv('./Music_Features_dataset(2).csv')
    model = load_model('MGD_reg_2.h5')
    dataset = data.drop(['filename','mfcc14','mfcc15','mfcc16','mfcc17','mfcc18','mfcc19','mfcc20'],axis=1)
    scaler=StandardScaler().fit(np.array(dataset.iloc[:, :-1], dtype = float))
    uploaded_data = pd.read_csv('./static/bin/UploadedMusic_Features_dataset.csv')
    uploaded__data = uploaded_data.drop(['filename','mfcc14','mfcc15','mfcc16','mfcc17','mfcc18','mfcc19','mfcc20'],axis=1)
    detect_audio = scaler.transform(np.array(uploaded__data, dtype = float))
    pred=model.predict_classes(detect_audio)
    print(pred)
    
    if(pred[0]==0):
        prediction="Blues"
    if(pred[0]==1):
        prediction="Classical"
    if(pred[0]==2):
        prediction="Country"
    if(pred[0]==3):
        prediction="Disco"
    if(pred[0]==4):
        prediction="Hiphop"
    if(pred[0]==5):
        prediction="Jazz"
    if(pred[0]==6):
        prediction="Metal"
    if(pred[0]==7):
        prediction="Pop"
    if(pred[0]==8):
        prediction="Reggae"
    if(pred[0]==9):
        prediction="Rock"
        
    audioName=f'/bin/{audioName}'   
    return render_template("analysis.html",prediction=prediction,audioName=audioName,col=category,
            fd=feature_data.split())
    
    
'''custom-analysis'''
@app.route("/custom-analysis",methods=["POST","GET"])
def custom_analysis():
    shutil.rmtree(bin_folder,ignore_errors = True)
    os.mkdir(bin_folder)
    if request.method == "POST":
        file = request.files['soundFile']
        cf = request.form.get('algo')
        file_name=file.filename
        file.save(f'./static/bin/{file_name}')
        return redirect(url_for("advanced_analysis",audioName=file_name,cf=cf))
    else:
        return render_template("custom-analysis.html")
        
'''advanced-analysis'''
@app.route("/audio-analysis/<audioName>/<cf>",methods=["POST","GET"])
def advanced_analysis(audioName,cf):
    
    '''generating dataset of uploaded audio file'''
    audio_data=f'./static/bin/{audioName}'
    file = open('./static/bin/UploadedMusic_Features_dataset.csv', 'w', newline='')
    category=['filename', 'chroma_stft', 'rmse', 'spectral_centroid', 'spectral_bandwidth', 
              'spectral_rolloff', 'zero_crossing_rate', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 
              'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12',
              'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20']
    
    '''extracting features'''
    with file:
        writer = csv.writer(file)
        writer.writerow(category)
    
    x, sr = librosa.load(audio_data, mono=True , duration=30)
    chroma_stft = librosa.feature.chroma_stft(y=x, sr=sr)
    rmse = librosa.feature.rmse(y=x)
    spectral_centroid = librosa.feature.spectral_centroid(y=x, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=x, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=x, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(x)
    mfcc = librosa.feature.mfcc(y=x, sr=sr)
    feature_data = f'{audioName} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spectral_centroid)} {np.mean(spectral_bandwidth)} {np.mean(spectral_rolloff)} {np.mean(zero_crossing_rate)}'    
    for m in mfcc:
        feature_data += f' {np.mean(m)}'
    file = open('./static/bin/UploadedMusic_Features_dataset.csv', 'a', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(feature_data.split())
        
       
    if cf == 'knn':
        algo='KNeighborsClassifier'
        
        ''' reading from our training dataset and fitting our test data to our knn model '''
        data = pd.read_csv('./Music_Features_dataset(2).csv')
        dataset = data.drop(['filename'],axis=1)
        y = LabelEncoder().fit_transform(dataset.iloc[:,-1])
        scaler=StandardScaler().fit(np.array(dataset.iloc[:, :-1], dtype = float))
        X=scaler.transform(dataset.iloc[:, :-1])
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42)
    
        ''' scaling our input data and applying our prediction '''
        uploaded_data = pd.read_csv('./static/bin/UploadedMusic_Features_dataset.csv')
        uploaded__data = uploaded_data.drop(['filename'],axis=1)
        detect_audio = scaler.transform(np.array(uploaded__data, dtype = float))
       
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train,y_train)
        
        pred = knn.predict(detect_audio[[0]]) 
        if(pred[0]==0):
            prediction="Blues"
        if(pred[0]==1):
            prediction="Classical"
        if(pred[0]==2):
            prediction="Country"
        if(pred[0]==3):
            prediction="Disco"
        if(pred[0]==4):
            prediction="Hiphop"
        if(pred[0]==5):
            prediction="Jazz"
        if(pred[0]==6):
            prediction="Metal"
        if(pred[0]==7):
            prediction="Pop"
        if(pred[0]==8):
            prediction="Reggae"
        if(pred[0]==9):
            prediction="Rock"
                    
        audioName=f'/bin/{audioName}'   
        return render_template("advanced-analysis.html",prediction=prediction,audioName=audioName,col=category,
                fd=feature_data.split(),algo=algo)
  
    elif cf == 'lr':
        algo='LogisticRegression'
        
        ''' reading from our training dataset and fitting our test data to our knn model '''
        data = pd.read_csv('./Music_Features_dataset(2).csv')
        dataset = data.drop(['filename'],axis=1)
        y = LabelEncoder().fit_transform(dataset.iloc[:,-1])
        scaler=StandardScaler().fit(np.array(dataset.iloc[:, :-1], dtype = float))
        X=scaler.transform(dataset.iloc[:, :-1])
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42)
    
        ''' scaling our input data and applying our prediction '''
        uploaded_data = pd.read_csv('./static/bin/UploadedMusic_Features_dataset.csv')
        uploaded__data = uploaded_data.drop(['filename'],axis=1)
        detect_audio = scaler.transform(np.array(uploaded__data, dtype = float))
       
        LR = LogisticRegression(C=0.01, solver='lbfgs', verbose=0 ,multi_class='auto').fit(X_train,y_train)
      
        pred = LR.predict(detect_audio[[0]]) 
        if(pred[0]==0):
            prediction="Blues"
        if(pred[0]==1):
            prediction="Classical"
        if(pred[0]==2):
            prediction="Country"
        if(pred[0]==3):
            prediction="Disco"
        if(pred[0]==4):
            prediction="Hiphop"
        if(pred[0]==5):
            prediction="Jazz"
        if(pred[0]==6):
            prediction="Metal"
        if(pred[0]==7):
            prediction="Pop"
        if(pred[0]==8):
            prediction="Reggae"
        if(pred[0]==9):
            prediction="Rock"
            
        audioName=f'/bin/{audioName}'   
        return render_template("advanced-analysis.html",prediction=prediction,audioName=audioName,col=category,
                fd=feature_data.split(),algo=algo)
   
    elif cf == 'svm':
         algo="SupportVectorMachines(kernel = rbf , poly)"
         
         ''' reading from our training dataset and fitting our test data to our knn model '''
         data = pd.read_csv('./Music_Features_dataset(2).csv')
         dataset = data.drop(['filename'],axis=1)
         y = LabelEncoder().fit_transform(dataset.iloc[:,-1])
         scaler=StandardScaler().fit(np.array(dataset.iloc[:, :-1], dtype = float))
         X=scaler.transform(dataset.iloc[:, :-1])
         X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42)
     
         ''' scaling our input data and applying our prediction '''
         uploaded_data = pd.read_csv('./static/bin/UploadedMusic_Features_dataset.csv')
         uploaded__data = uploaded_data.drop(['filename'],axis=1)
         detect_audio = scaler.transform(np.array(uploaded__data, dtype = float))
       
         rbf = svm.SVC(kernel='rbf').fit(X_train, y_train)
         poly = svm.SVC(kernel='poly', degree=1).fit(X_train, y_train)
         
         pred1 = poly.predict(detect_audio[[0]])
         if(pred1[0]==0):
            prediction1="Blues"
         if(pred1[0]==1):
            prediction1="Classical"
         if(pred1[0]==2):
            prediction1="Country"
         if(pred1[0]==3):
            prediction1="Disco"
         if(pred1[0]==4):
            prediction1="Hiphop"
         if(pred1[0]==5):
            prediction1="Jazz"
         if(pred1[0]==6):
            prediction1="Metal"
         if(pred1[0]==7):
            prediction1="Pop"
         if(pred1[0]==8):
            prediction1="Reggae"
         if(pred1[0]==9):
            prediction1="Rock"
            
         pred2 = rbf.predict(detect_audio[[0]])
         if(pred2[0]==0):
            prediction2="Blues"
         if(pred2[0]==1):
            prediction2="Classical"
         if(pred2[0]==2):
            prediction2="Country"
         if(pred2[0]==3):
            prediction2="Disco"
         if(pred2[0]==4):
            prediction2="Hiphop"
         if(pred2[0]==5):
            prediction2="Jazz"
         if(pred2[0]==6):
            prediction2="Metal"
         if(pred2[0]==7):
            prediction2="Pop"
         if(pred2[0]==8):
            prediction2="Reggae"
         if(pred2[0]==9):
            prediction2="Rock"
          
         audioName=f'/bin/{audioName}'   
         return render_template("advanced-analysis.html",prediction1=prediction1,prediction2=prediction2
                ,audioName=audioName,col=category,fd=feature_data.split(),algo=algo)
    
    elif cf == 'dt':
        algo = "DecisionTreeClassifier"
        
        ''' reading from our training dataset and fitting our test data to our knn model '''
        data = pd.read_csv('./Music_Features_dataset(2).csv')
        dataset = data.drop(['filename'],axis=1)
        y = LabelEncoder().fit_transform(dataset.iloc[:,-1])
        scaler=StandardScaler().fit(np.array(dataset.iloc[:, :-1], dtype = float))
        X=scaler.transform(dataset.iloc[:, :-1])
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42)
    
        ''' scaling our input data and applying our prediction '''
        uploaded_data = pd.read_csv('./static/bin/UploadedMusic_Features_dataset.csv')
        uploaded__data = uploaded_data.drop(['filename'],axis=1)
        detect_audio = scaler.transform(np.array(uploaded__data, dtype = float))
       
        musicTree = DecisionTreeClassifier(criterion="entropy", max_depth = 6)
        musicTree.fit(X_train,y_train)
        
        pred=musicTree.predict(detect_audio[[0]])
        if(pred[0]==0):
            prediction="Blues"
        if(pred[0]==1):
            prediction="Classical"
        if(pred[0]==2):
            prediction="Country"
        if(pred[0]==3):
            prediction="Disco"
        if(pred[0]==4):
            prediction="Hiphop"
        if(pred[0]==5):
            prediction="Jazz"
        if(pred[0]==6):
            prediction="Metal"
        if(pred[0]==7):
            prediction="Pop"
        if(pred[0]==8):
            prediction="Reggae"
        if(pred[0]==9):
            prediction="Rock"
                    
        audioName=f'/bin/{audioName}'   
        return render_template("advanced-analysis.html",prediction=prediction,audioName=audioName,col=category,
                fd=feature_data.split(),algo=algo)
        
    elif cf == 'ann':
        algo = "Artificial Neural Network"
        model = load_model('MGD_reg_2.h5')
        data = pd.read_csv('./Music_Features_dataset(2).csv')
        dataset = data.drop(['filename','mfcc14','mfcc15','mfcc16','mfcc17','mfcc18','mfcc19','mfcc20'],axis=1)
        scaler=StandardScaler().fit(np.array(dataset.iloc[:, :-1], dtype = float))
        uploaded_data = pd.read_csv('./static/bin/UploadedMusic_Features_dataset.csv')
        uploaded__data = uploaded_data.drop(['filename','mfcc14','mfcc15','mfcc16','mfcc17','mfcc18','mfcc19','mfcc20'],axis=1)
        detect_audio = scaler.transform(np.array(uploaded__data, dtype = float))
        pred=model.predict_classes(detect_audio)
        print(pred)
        
        if(pred[0]==0):
            prediction="Blues"
        if(pred[0]==1):
            prediction="Classical"
        if(pred[0]==2):
            prediction="Country"
        if(pred[0]==3):
            prediction="Disco"
        if(pred[0]==4):
            prediction="Hiphop"
        if(pred[0]==5):
            prediction="Jazz"
        if(pred[0]==6):
            prediction="Metal"
        if(pred[0]==7):
            prediction="Pop"
        if(pred[0]==8):
            prediction="Reggae"
        if(pred[0]==9):
            prediction="Rock"
                    
        audioName=f'/bin/{audioName}'   
        return render_template("advanced-analysis.html",prediction=prediction,audioName=audioName,col=category,
                fd=feature_data.split(),algo=algo)
        
        
    else:
         return render_template("custom-analysis.html")
        
   
'''playlist creation'''
@app.route("/create-playlist",methods=["POST","GET"])
def playlist():
    files=[]
    shutil.rmtree(bin_folder,ignore_errors = True)
    os.mkdir(bin_folder)
    if request.method == "POST":
        fileList=[]
        files = request.files.getlist("soundFile")
        for file in files:
            file_name=file.filename
            fileList.append(file_name)
            file.save(f'./static/bin/{file_name}')
            
        return redirect(url_for("playlist_generation",audioName=fileList))
    else:
        files=[]
        shutil.rmtree(bin_folder,ignore_errors = True)
        os.mkdir(bin_folder)
        return render_template("playlist.html")
    
    
'''playlist-genre-classification'''
@app.route("/playlist-generation/<audioName>")
def playlist_generation(audioName):
    audioName=audioName[2:-2].split("', '")
    ''' creating csv file for our input data and storing the extracted features '''
    file = open('./static/bin/UploadedMusic_Features_dataset.csv', 'w', newline='')
    category=['filename', 'chroma_stft', 'rmse', 'spectral_centroid', 'spectral_bandwidth',
              'spectral_rolloff', 'zero_crossing_rate', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4',
              'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12',
              'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20']
    with file:
        writer = csv.writer(file)
        writer.writerow(category)
    
    for audio in audioName:
        audio_data=f'./static/bin/{audio}'
        x, sr = librosa.load(audio_data, mono=True , duration=30)
        chroma_stft = librosa.feature.chroma_stft(y=x, sr=sr)
        rmse = librosa.feature.rmse(y=x)
        spectral_centroid = librosa.feature.spectral_centroid(y=x, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=x, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=x, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(x)
        mfcc = librosa.feature.mfcc(y=x, sr=sr)
        feature_data = f'{audio} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spectral_centroid)} {np.mean(spectral_bandwidth)} {np.mean(spectral_rolloff)} {np.mean(zero_crossing_rate)}'    
        for m in mfcc:
            feature_data += f' {np.mean(m)}'
        file = open('./static/bin/UploadedMusic_Features_dataset.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(feature_data.split())
        
    ''' reading from our training dataset and fitting our test data to our knn model '''    
    data = pd.read_csv('./Music_Features_dataset(2).csv')
    model = load_model('MGD_reg_2.h5')
    dataset = data.drop(['filename','mfcc14','mfcc15','mfcc16','mfcc17','mfcc18','mfcc19','mfcc20'],axis=1)
    scaler=StandardScaler().fit(np.array(dataset.iloc[:, :-1], dtype = float))
    uploaded_data = pd.read_csv('./static/bin/UploadedMusic_Features_dataset.csv')
    uploaded__data = uploaded_data.drop(['filename','mfcc14','mfcc15','mfcc16','mfcc17','mfcc18','mfcc19','mfcc20'],axis=1)
    detect_audio = scaler.transform(np.array(uploaded__data, dtype = float))
    pred=model.predict_classes(detect_audio)
    print(pred)
    
    
    prediction=[]
    for x in pred:
        if(x==0):
            p="Blues"
        if(x==1):
            p="Classical"
        if(x==2):
            p="Country"
        if(x==3):
            p="Disco"
        if(x==4):
            p="Hiphop"
        if(x==5):
            p="Jazz"
        if(x==6):
            p="Metal"
        if(x==7):
            p="Pop"
        if(x==8):
            p="Reggae"
        if(x==9):
            p="Rock"
        prediction.append(p)
        
    cat=[]
    for x in prediction:
        if x not in cat:
            cat.append(x)
    print(cat)
    string='/bin/'
    audioName=[string + x for x in audioName]        
    playList=[a for a in zip(prediction,audioName)]
    print(playList)
        
    return render_template("playlist-genre-classification.html",playList=playList, cat=cat)
    
    
    
if __name__ == "__main__":
    app.run()
