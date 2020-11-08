from flask import Flask, request, redirect, url_for, flash, jsonify, render_template
import numpy as np
import pickle as p
import json
import librosa
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing

app = Flask(__name__, static_folder='static')

@app.route('/index', methods=['GET'])
def home():
    return render_template('index.html')

def predict(path):
    features=[]  
    names=[]
    y,sr=librosa.load(path,mono=True,duration=30)
    length=librosa.get_duration(y=y,sr=sr) 
    print(length)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    features.append(np.mean(chroma_stft))
    features.append(np.var(chroma_stft))
    names.append('chroma_stft_mean')
    names.append('chroma_stft_var')
    
    rms = librosa.feature.rms(y=y)
    features.append(np.mean(rms))
    features.append(np.var(rms))
    names.append('rms_mean')
    names.append('rms_var')
    
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(np.mean(spec_cent))
    features.append(np.var(spec_cent))
    names.append('spectral_centroid_mean')
    names.append('spectral_centroid_var')
    
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.append(np.mean(spec_bw))
    features.append(np.var(spec_bw))
    names.append('spectral_bandwidth_mean')
    names.append('spectral_bandwidth_var')
    
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.append(np.mean(rolloff))
    features.append(np.var(rolloff))
    names.append('rolloff_mean')
    names.append('rolloff_var')

    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(np.mean(zcr))
    features.append(np.var(zcr))
    names.append('zero_crossing_rate_mean')
    names.append('zero_crossing_rate_var')
    
    har=librosa.effects.harmonic(y=y)
    features.append(np.mean(har))
    features.append(np.var(har))
    names.append('harmony_mean')
    names.append('harmony_var')
    
    per=librosa.effects.percussive(y=y)
    features.append(np.mean(per))
    features.append(np.var(per))
    names.append('perceptr_mean')
    names.append('perceptr_var')
    
    tempo=librosa.beat.tempo(y=y)
    features.append(np.mean(tempo))
    names.append('tempo')
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    c=1
    for i in mfcc:
        features.append(np.mean(i))
        features.append(np.var(i))
        names.append(f'mfcc{c}_mean')
        names.append(f'mfcc{c}_var')
        c+=1

    print('features extracted')
    print('number of features:',len(features))
    #print(features)
    datafr=[]
    datafr.append(features)
    dataframe=pd.DataFrame(datafr,columns=names)
    print('type of dataframe : ',type(dataframe))

    modelfile = 'final_xgboost_91_accuracy.pickle'
    model = p.load(open(modelfile, 'rb'))
    prediction = model.predict(dataframe)
    print(prediction[0])
    return prediction[0]

@app.route('/submit',methods=['POST'])
def submit():
    path=request.form['path']
    print('path of audio file:',path)   
    prediction=predict(path)
    return prediction


def findsimilarsongs(name,sim_df_names):
    series = sim_df_names[name].sort_values(ascending = False)
    series = series.drop(name)
    print("\n*******\nSimilar songs to ", name)
    d=dict(series)
    listtt=list(d.keys())
    listt=listtt[0:5]
    return listt

@app.route('/predict', methods=['GET'])
def renderSimilar():
    return render_template('predict.html')

@app.route('/similar',methods=['POST'])
def getsimilar():
    data = pd.read_csv(f'features_30_sec.csv', index_col='filename')
    labels = data[['label']]
    data = data.drop(columns=['length','label'])
    data_scaled=preprocessing.scale(data)
    print('Scaled data type:', type(data_scaled))
    similarity = cosine_similarity(data_scaled)
    print("Similarity shape:", similarity.shape)
    sim_df_labels = pd.DataFrame(similarity)
    sim_df_names = sim_df_labels.set_index(labels.index)
    sim_df_names.columns = labels.index
    name=request.form['path']
    name=name.split('/')[-1]
    print(name)
    songslist=findsimilarsongs(name,sim_df_names)
    return jsonify(songslist)


if __name__ == '__main__':
    app.run()