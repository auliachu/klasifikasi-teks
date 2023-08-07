import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def tokenisasi():
    data = pd.read_csv('depresi (3).csv')
    data.text = data.text.astype(str)
    text = data['text']
    tokenizer = Tokenizer(num_words=10000, oov_token='<UNK>')
    tokenizer.fit_on_texts(text)
    word_index = tokenizer.word_index

    return tokenizer

token = tokenisasi()

@st.cache_resource
def load_models():
    """
    @st.cache_resource decorator digunakan untuk menyimpan resource model.

    Fungsi load_models() akan membuat model FCDUG dan menerapkan weights dari file .h5 

    """

    model_LSTM = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=10000, output_dim=40),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
    ])
    

    model_LSTM.load_weights("LSTM.h5")

    return model_LSTM

model = load_models()


def preprocess_text(sentence):
    sentence = [sentence]
    print(sentence)

    # Preprocessing
    # melakukan tokenisasi
    #tokenizer = Tokenizer(num_words=10000, oov_token='<UNK>')
    #tokenizer.fit_on_texts(sentence)
    #tokenizer.fit_on_texts(sentence)
    sequences = token.texts_to_sequences(sentence)
    #sequences = tokenizer.fit_on_texts(sentence)
    print(sequences)
    padded = pad_sequences(sequences, maxlen=40, padding='post', truncating='post')
    print("Padded in preprocess ->", padded)

    return padded

def predict(text_predict):
    """
    @st.cache_data decorator berfungsi untuk caching / menyimpan data prediksi sementara

    Fungsi predict digunakan untuk melakukan prediksi data
    """
    padded = preprocess_text(text_predict)
    print(padded)
    prediction = int(np.argmax(model.predict(padded)))

    print("Prediction -> ", prediction)
    if prediction == 0:
        prediction = "Tidak Depresi"
    else:
        prediction = "Depresi"
    
    return prediction

def main():
    st.title("Text Classification Depression")
    st.subheader("Architecture used -> LSTM+Augmentation")

    menu = ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        with st.form(key='nlpForm'):
            raw_text = st.text_area("Enter Text Here")
            submit_button = st.form_submit_button(label='Analyze')

            col1  = st.columns(1)
            if submit_button:
                
                result = predict(raw_text)
                st.info("Results")
                st.write(result)
    
    else:
        st.subheader("About")

if __name__=='__main__':
    main()