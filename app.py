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
    y, sr = librosa.load(path, mono=True, duration=3)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    features.append(np.mean(chroma_stft))
    rmse = librosa.feature.rms(y=y)
    features.append(np.mean(rmse))
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(np.mean(spec_cent))
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.append(np.mean(spec_bw))
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.append(np.mean(rolloff))
    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(np.mean(zcr))
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    c=1
    for i in mfcc:
        features.append(np.mean(i))
        c+=1
    print('features extracted')
    print('number of features:',len(features))
    modelfile = 'xgboost_pred_model.pickle'
    model = p.load(open(modelfile, 'rb'))
    l=[]
    l+=features
    l+=features
    l+=features[0:6]
    list2=[]
    list2.append(l)
    list2.append(l)
    data=np.array(list2)
    prediction = np.array2string(model.predict(data))
    return jsonify(prediction)


if __name__ == '__main__':
    app.run()