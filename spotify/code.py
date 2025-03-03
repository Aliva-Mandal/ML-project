import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
from torch.utils.data import DataLoader, TensorDataset
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from sklearn.preprocessing import LabelEncoder

# Paths
DATASET_PATH = "dataset/spotify.csv"
MODEL_DIR = "models"
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
KERAS_MODEL_PATH = os.path.join(MODEL_DIR, "keras_model.h5")
PYTORCH_MODEL_PATH = os.path.join(MODEL_DIR, "pytorch_model.pth")
MAX_WORDS = 10000
MAX_SEQUENCE_LENGTH = 100

os.makedirs(MODEL_DIR, exist_ok=True)

def load_and_preprocess_data():
    df = pd.read_csv(DATASET_PATH)[['track_name', 'artist_name', 'lyrics']].dropna()
    df['song_id'] = LabelEncoder().fit_transform(df['track_name'] + " - " + df['artist_name'])
    
    tokenizer = Tokenizer(num_words=MAX_WORDS, lower=True, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['lyrics'])
    sequences = tokenizer.texts_to_sequences(df['lyrics'])
    padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="post")
    
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)
    
    return df, np.array(padded_sequences), df['song_id'].values

df, X, y = load_and_preprocess_data()

def train_keras_model():
    model = Sequential([
        Embedding(MAX_WORDS, 128, input_length=MAX_SEQUENCE_LENGTH),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dense(32, activation="relu"),
        Dense(len(np.unique(y)), activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X, y, epochs=5, batch_size=32, validation_split=0.2)
    model.save(KERAS_MODEL_PATH)

if not os.path.exists(KERAS_MODEL_PATH):
    train_keras_model()

class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x.long())
        x = self.transformer(x)
        x = torch.mean(x, dim=1)
        return self.fc(x)

def train_pytorch_model():
    dataset = TensorDataset(torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = TransformerModel(MAX_WORDS, 128, len(np.unique(y)))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(5):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} Loss: {loss.item()}")
    
    torch.save(model.state_dict(), PYTORCH_MODEL_PATH)

if not os.path.exists(PYTORCH_MODEL_PATH):
    train_pytorch_model()

# Load Tokenizer and Models
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

keras_model = tf.keras.models.load_model(KERAS_MODEL_PATH)

pytorch_model = TransformerModel(MAX_WORDS, 128, len(np.unique(y)))
pytorch_model.load_state_dict(torch.load(PYTORCH_MODEL_PATH))
pytorch_model.eval()

def predict_song(text_snippet):
    sequence = tokenizer.texts_to_sequences([text_snippet])
    padded_sequence = np.array(pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding="post"))
    
    keras_pred = np.argmax(keras_model.predict(padded_sequence), axis=1)[0]
    pytorch_pred = torch.argmax(pytorch_model(torch.tensor(padded_sequence, dtype=torch.long))).item()
    
    return keras_pred, pytorch_pred

if __name__ == "__main__":
    text = input("🎵 Enter a song lyric snippet: ")
    song_keras, song_pytorch = predict_song(text)
    print(f"\n🎶 TensorFlow Identified Song ID: {song_keras}")
    print(f"🎶 PyTorch Identified Song ID: {song_pytorch}")