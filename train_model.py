import os, numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

DATA_PATH = 'MP_Data'
actions = ['hello', 'thanks', 'iloveyou', 'Yes', 'help', 'stop']
sequence_length = 30
label_map = {label:i for i,label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for seq in os.listdir(os.path.join(DATA_PATH, action)):
        window = []
        for frame in range(sequence_length):
            window.append(np.load(os.path.join(DATA_PATH, action, seq, f"{frame}.npy")))
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)),
    LSTM(128, return_sequences=True, activation='relu'),
    LSTM(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(actions), activation='softmax')
])

model.compile('Adam','categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50)
model.save('action.h5')
