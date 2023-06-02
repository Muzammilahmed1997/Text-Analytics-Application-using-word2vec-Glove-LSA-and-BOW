# Importing important libraries

import numpy as np
import pandas as pd
import time
import re, string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re
import gensim
import time

from gensim.models import KeyedVectors

from sklearn.metrics.pairwise import cosine_similarity

import pickle
import streamlit as st
import warnings

warnings.filterwarnings('ignore')

start = time.time()
data = pd.read_csv('C:/Users/Muzammil/Desktop/TA_Assignment_1/TA_Data_Collection/sentences.csv')
data.shape




bow_vector= pickle.load(open('bow_vector', 'rb'))
bow_model= pickle.load(open('BOW_Model', 'rb'))

lsa_vector= pickle.load(open('lsa_vector', 'rb'))
lsa_model= pickle.load(open('LSA_Model', 'rb'))

customized_word2vec_vector = pickle.load(open('customized_vector', 'rb'))
customized_word2vec_model = pickle.load(open('customized_model', 'rb'))

word2vec_vector= pickle.load(open('W2V_Vector', 'rb'))
word2vec_model= pickle.load(open('word2vec_model', 'rb'))

Glove_vector= pickle.load(open('Glove_Pickle_vector', 'rb'))
Glove_model= pickle.load(open('Glove_Pickle_Model', 'rb'))



st.title("News Similarity App")

st.write('This Application will provide similar headlines')

## function to create sentence embedding

def sentence_embedding(data):
    words = data.split()
    embedding = np.zeros(300)
    count = 0
    for word in words:
        if word in word2vec_model.index_to_key:
            embedding += word2vec_model[word]
            count += 1
    if count > 0:
        embedding /= count
    return embedding

def glove_sentence_embedding(data):
    words = data.split()
    embedding = np.zeros(300)
    count = 0
    for word in words:
        if word in Glove_model.index_to_key:
            embedding += Glove_model[word]
            count += 1
    if count > 0:
        embedding /= count
    return embedding



def customized_sentence_embedding(data):
    words = data.split()
    embedding = np.zeros(50)
    count = 0
    for word in words:
        if word in customized_word2vec_model.index_to_key:
            embedding += customized_word2vec_model[word]
            count += 1
    if count > 0:
        embedding /= count
    return embedding

def lower_cased_punctuation_removal(text):
    for punc in string.punctuation:  #String.punctuation has all punctation marks
        text = text.replace(punc, '') #Reomval of that punctuation from each text
        s = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", text)
    return text.lower()  #Converting the text to lowercase

def query_preprocess_cusomized(text):
    processed_text = lower_cased_punctuation_removal(text)
    return processed_text


input_text = st.text_input("Enter any text")
# Display the user's input
st.write("You entered:", input_text)
input = []
input.append(input_text)


if st.checkbox("BOW Model"):
    st.write("BOW Button clicked!")


    input_text = bow_model.transform(input).toarray()

    cos_similarities = cosine_similarity(input_text , bow_vector)
    print(cos_similarities)
    print(cos_similarities.max())
    print(cos_similarities[0].argmax())
    print(cos_similarities[0].max())
    similar_sentences = cos_similarities[0].argsort()[-5:][::-1]

    for i in similar_sentences:
        print(data.sentence[i])
        #st.write(data.sentence[i])
        st.write(data.sentence[i])
    end = time.time()
    st.write(("Average Query Time", end - start))
    

elif st.checkbox("Customized Word2Vec Model"):
    st.write("Customized Word2Vec Button clicked!")
    #input_text = input_text.apply(gensim.utils.simple_preprocess)
    input_text = query_preprocess_cusomized(input_text)
    input_text = customized_sentence_embedding(input_text)
    input_text= input_text.reshape(1,50)

    cos_similarities = cosine_similarity(input_text , customized_word2vec_vector)
    print(cos_similarities)
    print(cos_similarities.max())
    print(cos_similarities[0].argmax())
    print(cos_similarities[0].max())
    similar_sentences = cos_similarities[0].argsort()[-5:][::-1]

    for i in similar_sentences:
        print(data.sentence[i])
        st.write(data.sentence[i])
    end = time.time()
    st.write(("Average Query Time", end - start))

elif st.checkbox("LSA Model"):
    st.write("LSA Button clicked!")

    input_text = bow_model.transform(input).toarray()

    cos_similarities = cosine_similarity(input_text , lsa_vector)
    print(cos_similarities)
    print(cos_similarities.max())
    print(cos_similarities[0].argmax())
    print(cos_similarities[0].max())
    similar_sentences = cos_similarities[0].argsort()[-5:][::-1]

    for i in similar_sentences:
        print(data.sentence[i])
        st.write(data.sentence[i])
    end = time.time()
    st.write(("Average Query Time", end - start))


elif st.checkbox("Word2Vec Model"):
    st.write("Word2Vec Button clicked!")

    input_text = sentence_embedding(input_text)
    input_text= input_text.reshape(1,300)

    cos_similarities = cosine_similarity(input_text , word2vec_vector)
    print(cos_similarities)
    print(cos_similarities.max())
    print(cos_similarities[0].argmax())
    print(cos_similarities[0].max())
    similar_sentences = cos_similarities[0].argsort()[-5:][::-1]

    for i in similar_sentences:
        print(data.sentence[i])
        st.write(data.sentence[i])
    end = time.time()
    st.write(("Average Query Time", end - start))


elif st.checkbox("Glove Model"):
    st.write("Glove Button clicked!")

    input_text = glove_sentence_embedding(input_text)
    input_text= input_text.reshape(1,300)

    cos_similarities = cosine_similarity(input_text , Glove_vector)
    print(cos_similarities)
    print(cos_similarities.max())
    print(cos_similarities[0].argmax())
    print(cos_similarities[0].max())
    similar_sentences = cos_similarities[0].argsort()[-5:][::-1]

    for i in similar_sentences:
        print(data.sentence[i])
        st.write(data.sentence[i])
    end = time.time()
    st.write(("Average Query Time", end - start))

