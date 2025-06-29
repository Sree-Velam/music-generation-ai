import os, pickle, numpy as np
from music21 import converter, instrument, note, chord, stream
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, Dense
from keras.utils import to_categorical

def extract_notes():
    notes = []
    for file in os.listdir("data"):
        if file.endswith(".mid"):
            midi = converter.parse(os.path.join("data", file))
            parts = instrument.partitionByInstrument(midi)
            notes_to_parse = parts.parts[0].recurse() if parts else midi.flat.notes
            for e in notes_to_parse:
                if isinstance(e, note.Note): notes.append(str(e.pitch))
                elif isinstance(e, chord.Chord):
                    notes.append('.'.join(str(n) for n in e.normalOrder))
    pickle.dump(notes, open("data/notes.pkl", "wb"))
    print(f"✅ Extracted {len(notes)} notes")

def train_model():
    notes = pickle.load(open("data/notes.pkl", "rb"))
    seq_len = 100
    pitches = sorted(set(notes))
    note2int = {p:i for i, p in enumerate(pitches)}

    seqs, outs = [], []
    for i in range(len(notes)-seq_len):
        seqs.append([note2int[n] for n in notes[i:i+seq_len]])
        outs.append(note2int[notes[i+seq_len]])

    X = np.reshape(seqs, (len(seqs), seq_len, 1)) / len(pitches)
    y = to_categorical(outs)

    model = Sequential([
        LSTM(256, input_shape=(seq_len,1), return_sequences=True),
        Dropout(0.3),
        LSTM(256),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(len(pitches), activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(X, y, epochs=30, batch_size=64)
    model.save("outputs/music_model.h5")
    print("✅ Model trained!")

def generate_music():
    notes = pickle.load(open("data/notes.pkl", "rb"))
    model = load_model("outputs/music_model.h5")
    pitches = sorted(set(notes))
    n_vocab = len(pitches)
    n2i = {p:i for i,p in enumerate(pitches)}
    i2n = {i:p for p,i in n2i.items()}

    seq_len = 100
    start = np.random.randint(0, len(notes) - seq_len - 1)
    pattern = [n2i[n] for n in notes[start:start+seq_len]]
    result = []

    for _ in range(200):
        inp = np.reshape(pattern, (1, seq_len, 1)) / n_vocab
        preds = model.predict(inp, verbose=0)
        idx = np.argmax(preds)
        result.append(i2n[idx])
        pattern.append(idx)
        pattern = pattern[1:]

    s = stream.Stream()
    offset = 0
    for p in result:
        if '.' in p or p.isdigit():
            ch = [note.Note(int(n)) for n in p.split('.')]
            for n in ch: n.storedInstrument = instrument.Piano()
            s.append(chord.Chord(ch).setOffset(offset))
        else:
            n2 = note.Note(p); n2.offset = offset; n2.storedInstrument = instrument.Piano()
            s.append(n2)
        offset += 0.5

    s.write('midi', fp='outputs/final_output.mid')
    print("🎶 Music generated: outputs/final_output.mid")

if __name__ == "__main__":
    os.makedirs('outputs', exist_ok=True)
    extract_notes()
    train_model()
    generate_music()
    print("✅ Done!")
