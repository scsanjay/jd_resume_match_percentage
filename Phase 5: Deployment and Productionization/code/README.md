# Commands to Run on the Instance

### To install the required applications
`sudo apt install python3-pip gunicorn python3-flask`

### To install the required python libraries
`pip install Flask numpy pandas pdfplumber matplotlib  fuzzywuzzy fuzzywuzzy[speedup] joblib gunicorn nltk gensim time scikit-learn==0.24.2  python-Levenshtein`

Need to download some nltk files,
`python -m nltk.downloader stopwords wordnet omw-1.4`

### To run the application in background
`gunicorn --bind 0.0.0.0:8080 -w 4 --limit-request-line 0 wsgi:app &`

### DEMO LINK 
http://13.234.90.146:8080/
