# import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdfplumber
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz
from joblib import load

def pdf2Text(filename):
    ''' load pdf and return the text'''
    text = ''
    # open the pdf file
    with pdfplumber.open(filename) as pdfObj:
        # loop over each page
        for page in pdfObj.pages:
            # get text from the page and concat
            text += page.extract_text()
    # return all texts
    return text

def concat(s):
    '''Concatenate words like "D A T A  S C I E N C E" to get "DATA SCIENCE"'''
    # add spaces at both end for better processing
    s = ' '+s+' '
    while True:
        # search if more than two alphabets are separated by space
        x = re.search(r"(\s[a-zA-Z]){2,}\s", s)
        if x==None:
            break
        # replace to get the concatenation
        s = s.replace(x.group(),' '+x.group().replace(' ','')+' ')
    return s

def preprocess_text(x, removeStopWords=False):
    # convert to lower case
    x = str(x).lower()
    # replace unusual quotes with '
    x = x.replace("′", "'").replace("’", "'")
    # replace new line with space
    x = x.replace("\n", " ")
    # concatenate
    x = concat(x)
    # remove links
    x = re.sub(r"http\S+", "", x)
    
    # convert education degrees like B.Tech or BTech to a specified form
    x = re.sub(r"\s+b[.]?[ ]?tech[(. /]{1}", " btech bachelor of technology ", x)
    x = re.sub(r"\s+m[.]?[ ]?tech[(. ]{1}", " mtech master of technology ", x)
    x = re.sub(r"\s+b[.]?[ ]?a[(. ]{1}", " ba bachelor of arts ", x)
    x = re.sub(r"\s+m[.]?[ ]?a[(. ]{1}", " ma master of arts ", x)
    x = re.sub(r"\s+b[.]?[ ]?sc[(. ]{1}", " bsc bachelor of science ", x)
    x = re.sub(r"\s+m[.]?[ ]?sc[(. ]{1}", " msc master of science ", x)
    x = re.sub(r"\s+b[.]?[ ]?e[(. ]{1}", " beng bachelor of engineering ", x)
    x = re.sub(r"\s+m[.]?[ ]?e[(. ]{1}", " meng master of engineering ", x)
    x = re.sub(r"\s+b[.]?[ ]?c[.]?[ ]?a[(. ]{1}", " bca bachelor of computer applications ", x)
    x = re.sub(r"\s+m[.]?[ ]?c[.]?[ ]?a[(. ]{1}", " mca master of computer applications ", x)
    x = re.sub(r"\s+b[.]?[ ]?b[.]?[ ]?a[(. ]{1}", " bba bachelor of business administration ", x)
    x = re.sub(r"\s+m[.]?[ ]?b[.]?[ ]?a[(. ]{1}", " mba master of business administration ", x)
    
    # convert skills with special symbols to words
    x = x.replace("c++", "cplusplus")
    x = x.replace("c#", "csharp")
    x = x.replace(".net", "dotnet")
    
    # replace non alpha numeric character with space
    x = re.sub('\W', ' ', x)
    
    # if remove stop words flag set then remove them
    z = []
    for i in x.split():
        if not (removeStopWords and i in stopwords.words('english')):
            # use lemmatizer to reduce the inflections
            lemmatizer = WordNetLemmatizer()
            i = lemmatizer.lemmatize(i)
            z.append(i)
    z = ' '.join(z)
    
    # strip white spaces
    z = z.strip()
    return z

# list of reputed colleges in India
reputed_colleges = ['bits', 'iit', 'bhu', 'nit', 'vit', 'anna', 'jadavpur', 'tiet', 'thapar', 'iisc', 'srm', 'dtu', 'iiit']
# check if is from reputed college
def is_from_reputed_college(x):
    x = x.split()
    for i in reputed_colleges:
        if i in x:
            return 1
    return 0

def feature_extract(data):
    '''extract features'''
    # number of words in resume
    data['resume_word_num'] = data.processed_resume.apply(lambda x: len(x.split()))
    # number of unique words in job description and resumes 
    data['total_unique_word_num'] = data.apply(lambda x: len(set(x.job_description.split()).union(set(x.processed_resume.split()))) ,axis=1)
    # number of common words in job description and resumes
    data['common_word_num'] = data.apply(lambda x: len(set(x.job_description.split()).intersection(set(x.processed_resume.split()))) ,axis=1)
    # number of common words divided by total number of unique words combined in both job description and resumes
    # data['common_word_ratio'] = data['common_word_num'] / data.apply(lambda x: len(set(x.job_description.split()).union(set(x.processed_resume.split()))) ,axis=1)
    # number of common words divided by minimum number of unique words between job description and resumes
    data['common_word_ratio_min'] = data['common_word_num'] / data.apply(lambda x: min(len(set(x.job_description.split())), len(set(x.processed_resume.split()))) ,axis=1) 
    # number of common words divided by maximum number of unique words between job description and resumes
    # data['common_word_ratio_max'] = data['common_word_num'] / data.apply(lambda x: max(len(set(x.job_description.split())), len(set(x.processed_resume.split()))) ,axis=1) 
    
    # Fuzz WRatio
    data["fuzz_ratio"] = data.apply(lambda x: fuzz.WRatio(x.job_description, x.processed_resume), axis=1)
    # Fuzz partial ratio
    data["fuzz_partial_ratio"] = data.apply(lambda x: fuzz.partial_ratio(x.job_description, x.processed_resume), axis=1)
    # Fuzz token set ratio
    data["fuzz_token_set_ratio"] = data.apply(lambda x: fuzz.token_set_ratio(x.job_description, x.processed_resume), axis=1)
    # Fuzz token sort ratio
    data["fuzz_token_sort_ratio"] = data.apply(lambda x: fuzz.token_sort_ratio(x.job_description, x.processed_resume), axis=1)
    
    # is fresher
    data['is_fresher'] = data.processed_resume.apply(lambda x: int('fresher' in x.split()))
    # from reputed college
    data['from_reputed_college'] = data.processed_resume.apply(lambda x: is_from_reputed_college(x))
    
    # fill na fields with 0
    data.fillna(0, inplace=True)
    return data

# Get cosine similarity and euclidean distance between two vectors
def cosine_euclidean(u, v):
    return np.array([np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), np.linalg.norm(u - v)])

def getAverageWord2Vec(sentence):
    # load word2vec model
    w2v_model = load('models/w2v_model.joblib')
    w2v_words = list(w2v_model.key_to_index)
    ''' get Average Word2Vec given a sentence'''
    # initialize sentence_vector to zeros
    sentence_vector = np.zeros(300)
    # count words in sentence
    count_words = 0
    # loop over each word
    for word in sentence.split():
        # if there is a vector for given word
        if word in w2v_words:
            # get the vector
            vector = w2v_model[word]
            # add the vectors
            sentence_vector = sentence_vector + vector
            # increment count
            count_words = count_words + 1
    if count_words != 0:
        # if the word count is not zero then divide by it to get the average
        sentence_vector /= count_words
    # return the avg word2vec
    return sentence_vector

