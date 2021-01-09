# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 11:09:40 2021
@author: Anirban Pal

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
import csv 

app = Flask(__name__)

app_root=os.path.dirname(os.path.abspath(__file__))
target=os.path.join(app_root,'static')
saved_file="audio.wav"


@app.route("/",methods=["POST","GET"])
def home():
    if request.method == "POST":
        file = request.files['soundFile']
        file_name=file.filename
        destination='/'.join([target,saved_file])
        file.save(destination)
        return redirect(url_for("analysis",audioName=file_name))
    else:
        return render_template("index.html")

@app.route("/about")
def about():
        return render_template("about.html")
    
@app.route("/<audioName>",methods=["POST","GET"])
def analysis(audioName):
    print(audioName)
    ''' creating csv file for our input data and storing the extracted features '''
    audio_data='/'.join([target,saved_file])
    file = open('./static/UploadedMusic_Features_dataset.csv', 'w', newline='')
    category=['filename', 'chroma_stft', 'rmse', 'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'zero_crossing_rate', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20']
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
    file = open('./static/UploadedMusic_Features_dataset.csv', 'a', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(feature_data.split())
        
    ''' reading from our training dataset and fitting our test data to our knn model '''    
    data = pd.read_csv('./Music_Features_dataset(2).csv')
    dataset = data[data['label'].isin(['jazz', 'metal','classical','blues','pop'])].drop(['filename'],axis=1)
    y = LabelEncoder().fit_transform(dataset.iloc[:,-1])
    scaler=StandardScaler().fit(np.array(dataset.iloc[:, :-1], dtype = float))
    X=scaler.transform(dataset.iloc[:, :-1])
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train,y_train)
    
    ''' scaling our input data and applying our prediction '''
    uploaded_data = pd.read_csv('./static/UploadedMusic_Features_dataset.csv')
    uploaded_dataset = uploaded_data.drop(['filename'],axis=1)
    detect_audio = scaler.transform(np.array(uploaded_dataset, dtype = float))
   
    pred = knn.predict(detect_audio[[0]]) 
    
    if(pred[0]==0):
        prediction="Blues"
    if(pred[0]==1):
        prediction="Classical"
    if(pred[0]==2):
        prediction="Jazz"
    if(pred[0]==3):
        prediction="Metal"
    if(pred[0]==4):
        prediction="Pop"
        
    
    return render_template("analysis.html",audioName=audioName,fd=feature_data.split(),col=category,prediction=prediction)

if __name__ == "__main__":
    app.run()