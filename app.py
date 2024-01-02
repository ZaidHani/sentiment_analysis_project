# import libraires
import streamlit as st
import pickle
import re

from nltk.corpus import stopwords
import nltk
from nltk.stem import SnowballStemmer

nltk.download('punkt')
nltk.download('stopwords')

# make a stemmer
stemmer = SnowballStemmer('english')

# load the model
with open('model50k.pkl', 'rb') as f:
    cv, nv = pickle.load(f)

# nlp functions
def nlp_functions(content):
    review = re.sub('[^a-zA-Z]', ' ', content.lower())
    review = nltk.word_tokenize(review)
    review = [word for word in review if word not in stopwords.words('english')]
    review = [stemmer.stem(word) for word in review]
    review = ' '.join(review)
    return review

# title
st.title("Sentiment Analysis by Zaid")

# input from user
text = st.text_input('Write a review')

# output to be predicted
clean_text = nlp_functions(text)
test_bow = cv.transform([clean_text]).toarray()
rating = nv.predict(test_bow)

if text:
    st.markdown(f"Predicted Rating: {rating[0]}")