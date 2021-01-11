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
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import classification_report
import csv 

app = Flask(__name__)

app_root=os.path.dirname(os.path.abspath(__file__))
target=os.path.join(app_root,'static')
saved_file="audio.wav"
saved_csv="UploadedMusic_Features_dataset.csv"
saved_img1=""
saved_img2=""

@app.route("/",methods=["POST","GET"])
def home():  
    if request.method == "POST":
        file = request.files['soundFile']
        file_name=file.filename
        destination=f'./static/{file_name}'
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
    audio_data=f'./static/{audioName}'
    file = open('./static/UploadedMusic_Features_dataset.csv', 'w', newline='')
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
          
    return render_template("analysis.html",audioName=audioName,fd=feature_data.split(),
                           col=category,prediction=prediction)

    
'''custom-analysis'''
@app.route("/custom-analysis",methods=["POST","GET"])
def custom_analysis():
    if request.method == "POST":
        file = request.files['soundFile']
        s_rate = request.form.get('sr')
        cf = request.form.get('algo')
        fl = request.form.getlist('featureList')
        file_name=file.filename
        file.save(f'./static/{file_name}')
        print(s_rate)
        print(cf)
        print(file_name)
        print(fl)
        return redirect(url_for("advanced_analysis",audioName=file_name,s_rate=s_rate,cf=cf,fl=fl))
    else:
        return render_template("custom-analysis.html")
        
'''advanced-analysis'''
@app.route("/audio-analysis/<audioName>/<s_rate>/<cf>/<fl>",methods=["POST","GET"])
def advanced_analysis(audioName,s_rate,cf,fl):
    print(cf)
    
    '''generating dataset of uploaded audio file'''
    audio_data=f'./static/{audioName}'
    file = open('./static/UploadedMusic_Features_dataset.csv', 'w', newline='')
    custom_category=fl[2:-2].split("', '")
    custom_category.append('label')
    srate=int(s_rate)
    print(custom_category)
    category=['filename', 'chroma_stft', 'rmse', 'spectral_centroid', 'spectral_bandwidth', 
              'spectral_rolloff', 'zero_crossing_rate', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 
              'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12',
              'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20']
    with file:
        writer = csv.writer(file)
        writer.writerow(category)
    
    x, sr = librosa.load(audio_data, mono=True , duration=30 , sr=srate)
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
        
    
    '''image generate'''
    plt.figure(figsize=(14, 5))
    plt.plot(x)
    plt.savefig(f'./static/audio_waveplot_{audioName}.png')
    saved_img1=f'audio_waveplot_{audioName}.png'
    cmap = plt.get_cmap('plasma')
    plt.figure(figsize=(14,5))
    plt.specgram(x, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
    plt.axis('off');
    plt.savefig(f'./static/audio_specgram_{audioName}.png')
    saved_img2=f'audio_specgram_{audioName}.png'
        
    ''' reading from our training dataset and fitting our test data to our knn model '''    
    data = pd.read_csv('./Music_Features_dataset(2).csv',skipinitialspace=True,usecols=custom_category)
    dataset = data[data['label'].isin(['jazz', 'metal','classical','blues','pop'])]
    y = LabelEncoder().fit_transform(dataset.iloc[:,-1])
    scaler=StandardScaler().fit(np.array(dataset.iloc[:, :-1], dtype = float))
    X=scaler.transform(dataset.iloc[:, :-1])
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20)
    
    ''' scaling our input data and applying our prediction '''
    uploaded_data = pd.read_csv('./static/UploadedMusic_Features_dataset.csv',
                                skipinitialspace=True,usecols=fl[2:-2].split("', '"))
    detect_audio = scaler.transform(np.array(uploaded_data, dtype = float))
       
    if cf == 'knn':
        algo='KNeighborsClassifier'
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train,y_train)
        
        pred_dummy = knn.predict(X_test)
        report=classification_report(y_test,pred_dummy,output_dict=True)
        df = pd.DataFrame(report).transpose()
        print(df)
        
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
            
        p=uploaded_data[0:1]
        t=[list(x) for x in p.values]
        user_feature_values=t[0]
        return render_template("advanced-analysis.html",prediction=prediction,audioName=audioName,
                f_table=[df.to_html(classes='tab')],titles=df.columns.values,
                fd=user_feature_values,col=fl[2:-2].split("', '"),
                saved_img1=saved_img1,saved_img2=saved_img2,sRate=srate,algo=algo)
  
    elif cf == 'lr':
        algo='LogisticRegression'
        LR = LogisticRegression(C=0.01, solver='lbfgs', verbose=0 ,multi_class='auto').fit(X_train,y_train)
      
        pred_dummy = LR.predict(X_test)
        report=classification_report(y_test,pred_dummy,output_dict=True)
        df = pd.DataFrame(report).transpose()
        print(df)
        
        pred = LR.predict(detect_audio[[0]]) 
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
            
        p=uploaded_data[0:1]
        t=[list(x) for x in p.values]
        user_feature_values=t[0]
        return render_template("advanced-analysis.html",prediction=prediction,audioName=audioName,
                f_table=[df.to_html(classes='tab')],titles=df.columns.values,
                fd=user_feature_values,col=fl[2:-2].split("', '"),
                saved_img1=saved_img1,saved_img2=saved_img2,sRate=srate,algo=algo)
   
    elif cf == 'svm':
         algo="SupportVectorMachines(kernel = rbf , poly)"
         rbf = svm.SVC(kernel='rbf').fit(X_train, y_train)
         poly = svm.SVC(kernel='poly', degree=1).fit(X_train, y_train)
    
         poly_pred = poly.predict(X_test)
         rbf_pred = rbf.predict(X_test)
         report1=classification_report(y_test, poly_pred,output_dict=True)
         df1 = pd.DataFrame(report1).transpose()
         print(df1)
         report2=classification_report(y_test, rbf_pred,output_dict=True)
         df2 = pd.DataFrame(report2).transpose()
         print(df2)
         
         pred1 = poly.predict(detect_audio[[0]])
         if(pred1[0]==0):
            prediction1="Blues"
         if(pred1[0]==1):
            prediction1="Classical"
         if(pred1[0]==2):
            prediction1="Jazz"
         if(pred1[0]==3):
            prediction1="Metal"
         if(pred1[0]==4):
            prediction1="Pop"
            
         pred2 = rbf.predict(detect_audio[[0]])
         if(pred2[0]==0):
            prediction2="Blues"
         if(pred2[0]==1):
            prediction2="Classical"
         if(pred2[0]==2):
            prediction2="Jazz"
         if(pred2[0]==3):
            prediction2="Metal"
         if(pred2[0]==4):
            prediction2="Pop"
          
         p=uploaded_data[0:1]
         t=[list(x) for x in p.values]
         user_feature_values=t[0]
         return render_template("advanced-analysis.html",algo=algo,prediction1=prediction1,
                    prediction2=prediction2,audioName=audioName,
                    f_table1=[df1.to_html(classes='tab')],titles1=df1.columns.values,
                    f_table2=[df2.to_html(classes='tab')],titles2=df2.columns.values,
                    fd=user_feature_values,col=fl[2:-2].split("', '"),saved_img1=saved_img1,
                    saved_img2=saved_img2,sRate=srate)
    
    elif cf == 'dt':
        algo = "DecisionTreeClassifier"
        musicTree = DecisionTreeClassifier(criterion="entropy", max_depth = 6)
        musicTree.fit(X_train,y_train)
        
        predTree = musicTree.predict(X_test)
        acc= metrics.accuracy_score(y_test, predTree)
        
        pred=musicTree.predict(detect_audio[[0]])
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
            
        p=uploaded_data[0:1]
        t=[list(x) for x in p.values]
        user_feature_values=t[0]
        
        dot_data = StringIO()
        filename = f'musictree_{audioName}.png'
        featureNames = fl[2:-2].split("', '")
        targetNames = dataset["label"].unique().tolist()
        out=tree.export_graphviz(musicTree,feature_names=featureNames, out_file=dot_data, filled=True, 
                                 special_characters=True,rotate=False)  
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
        graph.write_png(f'./static/{filename}')
        img = mpimg.imread(f'./static/{filename}')
        plt.figure(figsize=(200, 400))
        plt.imshow(img,interpolation='nearest')
        
        return render_template("advanced-analysis.html",prediction=prediction,audioName=audioName,
                acc=acc,fd=user_feature_values,col=fl[2:-2].split("', '"),
                saved_img1=saved_img1,saved_img2=saved_img2,sRate=srate,algo=algo,saved_img3=filename)
        
    else:
         return render_template("custom-analysis.html")
        
    
    
if __name__ == "__main__":
    app.run()