import streamlit as st
import tensorflow as tf

# Streamlit App

def predict_response(model, vectorizer, input_text, max_length=20):
    # Preprocess input text
    input_text = standardize_text(input_text)

    # Vectorize the input text
    input_seq = vectorizer([input_text]).numpy()[0].tolist()
    input_seq = pad_sequences([input_seq], maxlen=max_seq_len, padding='pre')

    # Initialize the decoder input with a start token or a padding token
    #decoder_input = [vectorizer.word_index['<start>']]  # Assuming '<start>' is in vocabulary
    # Alternatively, if '<start>' is not in vocabulary:
    decoder_input = [0]  # Assuming 0 is the index of the padding token
    
    # Change here: Padding with max_seq_len instead of max_seq_len - 1
    decoder_input = pad_sequences([decoder_input], maxlen=max_seq_len, padding='post') 

    # Generate the response, token by token
    output_text = ""
    for _ in range(max_length):
        predictions = model.predict([input_seq, decoder_input])
        predicted_id = np.argmax(predictions[0][-1])
        output_word = vectorizer.get_vocabulary()[predicted_id]

        if output_word == '<end>':
            break

        output_text += " " + output_word
        decoder_input = [predicted_id]
        decoder_input = pad_sequences([decoder_input], maxlen=max_seq_len-1, padding='post')

        # Early stopping if the same word is repeated consecutively
        if len(output_text.split()) > 1 and output_text.split()[-1] == output_text.split()[-2]:
            break

    return output_text.strip()

    def standardize_text(text):
        """Lowercase and remove special characters from text."""
        text = text.lower()  # Lowercase
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special chars
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')  # Unicode to ASCII
    return text

    vectorizer = TextVectorization(
        max_tokens=10000,  # Vocabulary size
        output_mode="int",  # Output numerical indices
        output_sequence_length = 10, # Set output sequence length
)

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