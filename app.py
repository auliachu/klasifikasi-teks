import streamlit as st
import pickle
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

""" bingung gimana cara pakai modelnya
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=10000, output_dim=40),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])
"""
@st.cache_resource
def load_models():
    """
    @st.cache_resource decorator digunakan untuk menyimpan resource model.

    Fungsi load_models() akan membuat model FCDUG dan menerapkan weights dari file .h5 

    """
    model.load_weights("LSTM.h5")

    return model

model = load_models()




def main():
    st.title("Text Classification Depression")
    st.subheader("Hello Streamlit")

    menu = ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        with st.form(key='nlpForm'):
            raw_text = st.text_area("Enter Text Here")
            submit_button = st.form_submit_button(label='Analyze')
            
            #melakukan tokenisasi
            tokenizer = Tokenizer(num_words=10000, oov_token='<UNK>')
            tokenizer.fit_on_texts(submit_button) 

            #melakukan padding sequence
            def get_sequences(tokenizer, tweets):
                sequences = tokenizer.texts_to_sequences(tweets)
                padded_sequences = pad_sequences(sequences, truncating='post', maxlen=40, padding='post')
                return padded_sequences 

            train_sequences = get_sequences(tokenizer, submit_button)


            #model

            #prediction
            @st.cache_data()
            def predict(text_predict):
                """
                @st.cache_data decorator berfungsi untuk caching / menyimpan data prediksi sementara

                Fungsi predict digunakan untuk melakukan prediksi data
                """
                image = preprocess_image(image_predict)
                prediction = model.predict(image)
                
                return prediction

        col1,col2 = st.columns(2)
        if submit_button:
            with col1:
                st.info("Results")
            with col2:
                st.info("Word Classification")
    
    else:
        st.subheader("About")

if __name__=='__main__':
    main()