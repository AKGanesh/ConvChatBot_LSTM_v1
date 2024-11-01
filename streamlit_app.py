import streamlit as st
import tensorflow as tf

# Streamlit App
def main():
    st.title("Language Model App")

    user_input = st.text_input("Enter your prompt:")
    if st.button("Generate"):
        generated_text = predict_response(model, vectorizer, user_input)
        st.write(generated_text)

if __name__ == '__main__':
    # Load the saved model
    model = tf.keras.models.load_model('convai_v1.h5')
    main()