import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from joblib import load
from helper import *
import flask
import os
import numpy as np
import time

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 3 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'resume' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['resume']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # path of the resume on server
        filePath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filePath)
    else:
        flash('File format not allowed.')
        return redirect(request.url)

    startTime = time.time()
    # load preprocessed jd
    jd_processed = load('models/jd_processed.joblib')
    # load and get the text from pdf
    res = pdf2Text(filePath)
    # preprocess resume
    res_processed = preprocess_text(res, removeStopWords=True)

    # create df
    data = pd.DataFrame({'job_description': [jd_processed], 'processed_resume': [res_processed]})
    # extract features
    data_feature = feature_extract(data)
    data_feature1 = data_feature.copy()
    data_feature2 = data_feature.copy()

    # load BoW representation of jd
    bow_jd = load('models/bow_jd.joblib')
    # load BoW model
    vectorizer = load('models/vectorizer.joblib')
    # get BoW representation of resume
    bow_resume = vectorizer.transform(data_feature.processed_resume.values).toarray()
    # get cosine similarity and euclidean distance features
    cosine_euclidean_data = np.array([cosine_euclidean(bow_jd[i], bow_resume[i]) for i in range(len(bow_resume))])
    data_feature1[["cosine_similarity", "euclidean_distance"]] = cosine_euclidean_data

    # create input for BoW based base model
    X_bow_1 = data_feature1.drop(columns=['job_description', 'processed_resume'])
    X_bow_2 = pd.DataFrame(bow_jd, columns=['bow_jd_'+str(i) for i in range(1, bow_jd.shape[1]+1)])
    X_bow_3 = pd.DataFrame(bow_resume, columns=['bow_resume_'+str(i) for i in range(1, bow_resume.shape[1]+1)])
    X_bow = pd.concat([X_bow_1, X_bow_2, X_bow_3], axis=1)
    
    # get selected features after forward feature selection
    with open('models/selected_features1.npy', 'rb') as f:
        features1 = np.load(f, allow_pickle=True)
    # filter input according to forward feature selection
    X_bow = X_bow[features1]
    
    # load w2v representation of jd
    w2v_jd = load('models/w2v_jd.joblib')
    # get w2v representation of resume
    w2v_resume = np.array([getAverageWord2Vec(data_feature2.processed_resume.values[0])])
    # get cosine similarity and euclidean distance features
    cosine_euclidean_data = np.array([cosine_euclidean(w2v_jd[i], w2v_resume[i]) for i in range(len(w2v_resume))])
    data_feature2[["cosine_similarity", "euclidean_distance"]] = cosine_euclidean_data

    # create input for Word2Vec based base model
    X_w2v_1 = data_feature2.drop(columns=['job_description', 'processed_resume'])
    X_w2v_2 = pd.DataFrame(w2v_jd, columns=['w2v_jd_'+str(i) for i in range(1, w2v_jd.shape[1]+1)])
    X_w2v_3 = pd.DataFrame(w2v_resume, columns=['w2v_resume_'+str(i) for i in range(1, w2v_resume.shape[1]+1)])
    X_w2v = pd.concat([X_w2v_1, X_w2v_2, X_w2v_3], axis=1)
    
    # get selected features after forward feature selection
    with open('models/selected_features2.npy', 'rb') as f:
        features2 = np.load(f, allow_pickle=True)
    # filter input according to forward feature selection
    X_w2v = X_w2v[features2]

    # load standard scaler for bow
    scaler1 = load('models/scaler1.joblib')
    X_bow = scaler1.transform(X_bow)
    # load standard scaler for w2v
    scaler2 = load('models/scaler2.joblib')
    X_w2v = scaler2.transform(X_w2v)

    # load first base model
    svr_model_linear_1 = load('models/svr_model_linear_1.joblib')
    # load second base model
    lr_model_2 = load('models/lr_model_2.joblib')

    # get the ouputs of base model
    X_ensemble = pd.DataFrame({'svr_linear_bow':svr_model_linear_1.predict(X_bow), 
                                'linear_reg_w2v':lr_model_2.predict(X_w2v)})
    # load standard scaler for ensemble
    scaler3 = load('models/scaler3.joblib')
    X_ensemble = scaler3.transform(X_ensemble)

    # load meta regressor
    knn_model_meta = load('models/knn_model_meta.joblib')

    # predict the o/p
    prediction = np.round(knn_model_meta.predict(X_ensemble), 2)
    endTime = time.time()

    return flask.render_template('predict.html', executiontime=np.round(endTime-startTime, 2), prediction=str(prediction[0])+'%')

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
