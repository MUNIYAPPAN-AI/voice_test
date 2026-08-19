import os
import numpy as np
import tensorflow as tf
import speech_recognition as sr 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

print("Initializing Big Vocabulary RNN Model...")

training_sentences = [
    "yes", "yeah", "yep", "sure", "of course", "absolutely", "do it", "proceed", "go ahead",
    "correct", "right", "good", "great", "awesome", "perfect", "ok", "okay", "fine",
    "confirm", "approved", "allow", "accept", "yes please", "make it happen", "run it",

    "no", "nay", "nope", "not at all", "never", "stop", "cancel", "abort", "reject",
    "wrong", "bad", "don't", "dont do it", "deny", "block", "disapprove", "quit",
    "no way", "terminate", "hold", "pause", "leave it", "false", "negative"
]

training_labels = np.array([
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
])

vocab_size = 500
max_length = 5

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(training_sentences)

sequences = tokenizer.texts_to_sequences(training_sentences)
padded_training = pad_sequences(sequences, maxlen=max_length, padding='post')

model = Sequential([
    Embedding(vocab_size, 16, input_length=max_length),
    SimpleRNN(16),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(padded_training, training_labels, epochs=50, verbose=0)
print("Dynamic RNN Brain Training Complete!")

def record_voice_command():
    recognizer = sr.Recognizer()
    
  
    with sr.Microphone() as source:
        print("\n[Status: Adjusting for background noise... Please wait 1 second]")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        
        print("\n Speak Now! (speak...)")
        print("Listening...")
        
        try:
            
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            print("Processing your voice...")
            
            
            text = recognizer.recognize_google(audio)
            return text
            
        except sr.WaitTimeoutError:
            return "ERROR: No speech detected (you speak now)"
        except sr.UnknownValueError:
            return "ERROR: Could not understand audio (can't understand)"
        except sr.RequestError as e:
            return f"ERROR: Service issue; {e}"

try:
    print("\n" + "="*50)
    print("Speak ANY Confirmation or Denial word freely now!")
    print("Examples: 'Yeah go ahead', 'Nope stop', 'Absolutely right', 'Never do that'")
    print("="*50 + "\n")

    captured_text = record_voice_command()

    if "ERROR" in captured_text:
        print(f" Blockage: {captured_text}")
    else:
        print(f" Captured Voice Text: \"{captured_text}\"")

        test_seq = tokenizer.texts_to_sequences([captured_text.lower()])
        test_padded = pad_sequences(test_seq, maxlen=max_length, padding='post')

        prediction = model.predict(test_padded, verbose=0)
        score = prediction[0][0]

        print("\n" + "="*40)
        print(f"RNN Polarity Weight Score: {score:.4f}")

        if score > 0.5:
            print(" RESULT: POSITIVE INTENT DETECTED! (Yes / Confirm)")
        else:
            print(" RESULT: NEGATIVE INTENT DETECTED! (No / Cancel)")
        print("="*40 + "\n")

except Exception as e:
    print("Execution halt:", e)