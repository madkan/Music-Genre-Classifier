from flask import Flask, request, redirect, url_for, flash, jsonify, render_template
import numpy as np
import pickle as p
import json
import librosa

app = Flask(__name__)

@app.route('/index', methods=['GET'])
def home():
    return render_template('index.html')



@app.route('/submit',methods=['POST'])
def submit():
    path=request.form['path']
    print('path of audio file:',path)
    features=[]  
    y,sr=librosa.load(path,mono=True,duration=3)
    
    length=librosa.get_duration(y=y,sr=sr)
    print(length)
    features.append(length)
    
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    features.append(np.mean(chroma_stft))
    features.append(np.var(chroma_stft))
    
    rms = librosa.feature.rms(y=y)
    features.append(np.mean(rms))
    features.append(np.var(rms))
    
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(np.mean(spec_cent))
    features.append(np.var(spec_cent))
    
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.append(np.mean(spec_bw))
    features.append(np.var(spec_bw))
    
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.append(np.mean(rolloff))
    features.append(np.var(rolloff))
    
    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(np.mean(zcr))
    features.append(np.var(zcr))
    
    har=librosa.effects.harmonic(y=y)
    features.append(np.mean(har))
    features.append(np.var(har))
    
    per=librosa.effects.percussive(y=y)
    features.append(np.mean(per))
    features.append(np.var(per))
    
    tempo=librosa.beat.tempo(y=y)
    features.append(np.mean(tempo))
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    for i in mfcc:
        features.append(np.mean(i))
        features.append(np.var(i))
    print('features extracted')
    print('number of features:',len(features))
    modelfile = 'xgboost_pred_model.pickle'
    model = p.load(open(modelfile, 'rb'))
    l=[]
    l.append(features)
    l.append(features)
    data=np.array(l)
    prediction = np.array2string(model.predict(data))
    return jsonify(prediction)


if __name__ == '__main__':
    app.run()