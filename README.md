# Music-Genre-Classifier
A simple website using flask to predict genre of a song. The machine learning model is trained on GTZAN 1000 songs dataset and uses XGboost Algorithm. Notebooks folder contains the jupyter notebook files containing feature extraction, visualizations of extraction and accuracy comparison across the algorithms SVM, Random Forest, Xgboost, Decsion Tree, K-nearest Neighbours. The highest accuracy of 91% was obtained for xgboost.

to run the application:
- create virtual environment
- $ . venv/bin/activate
- $ pip3 install flask
- $ export FLASK_APP=app.py (entry point of the app)
- $ flask run
- the route of the index page is localhost:5000/index


