import os
import re
import pickle
import threading
from collections import defaultdict, Counter
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================= TensorFlow (optional) =================
# Force CPU-only to prevent GPU JIT compilation errors (libdevice missing).
# The LSTM still works correctly on CPU; only speed differs.
import os as _os
_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Embedding, LSTM, Dense
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ================= CONFIG =================
CORPUS_DIR = "corpus"
LSTM_MODEL = "lstm.h5"
TOKENIZER_FILE = "tokenizer.pkl"

N = 3
TOP_K = 3

# ================= TEXT CLEANING =================
def clean_text(text: str) -> str:
    text = text.lower()
    # Keep only a-z and basic punctuation if needed, currently only a-z and space
    text = re.sub(r"[^a-z\s']", " ", text) # allow apostrophes optionally
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def tokenize(text: str):
    return clean_text(text).split()

# ================= LOAD MULTI-CORPUS =================
def load_all_corpus_files():
    texts = []
    if not os.path.exists(CORPUS_DIR):
        return texts

    for file in os.listdir(CORPUS_DIR):
        if file.endswith(".txt"):
            path = os.path.join(CORPUS_DIR, file)
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = clean_text(line)
                    if len(line.split()) >= 3:
                        texts.append(line)
    return texts

# ================= DATA CLASS =================
@dataclass
class Prediction:
    word: str
    probability: float

    def to_dict(self):
        return {"word": self.word, "probability": self.probability}

# ================= NGRAM =================
class NGramPredictor:
    def __init__(self):
        self.ngrams = defaultdict(Counter)
        self.vocab = Counter()
        self.parent = None

    def train(self, texts):
        for text in texts:
            tokens = tokenize(text)
            self.vocab.update(tokens)
            for i in range(len(tokens)):
                for n in range(1, N + 1):
                    if i + n <= len(tokens):
                        ctx = tuple(tokens[i:i + n - 1])
                        self.ngrams[ctx][tokens[i + n - 1]] += 1

    def learn_sentence(self, text):
        tokens = tokenize(text)
        self.vocab.update(tokens)
        for i in range(len(tokens)):
            for n in range(1, N + 1):
                if i + n <= len(tokens):
                    ctx = tuple(tokens[i:i + n - 1])
                    self.ngrams[ctx][tokens[i + n - 1]] += 1

        if self.parent:
            self.parent.new_samples += 1

    def predict(self, context):
        tokens = tokenize(context)
        for n in range(N, 0, -1):
            ctx_len = n - 1
            ctx = tuple(tokens[-ctx_len:]) if ctx_len > 0 else ()
            if ctx in self.ngrams:
                total = sum(self.ngrams[ctx].values())
                return [
                    Prediction(w, c / total)
                    for w, c in self.ngrams[ctx].most_common(TOP_K)
                ]
        return []

    def complete(self, partial):
        partial = clean_text(partial)
        matches = [(w, c) for w, c in self.vocab.items() if w.startswith(partial)]
        total = sum(c for _, c in matches) or 1
        matches.sort(key=lambda x: x[1], reverse=True)
        return [
            Prediction(w, c / total)
            for w, c in matches[:TOP_K]
        ]

# ================= MAIN PREDICTOR =================
class WordPredictor:
    def __init__(self):
        self.ngram = NGramPredictor()
        self.ngram.parent = self

        self.lstm = None
        self.tokenizer = None
        self.is_retraining = False
        self.new_samples = 0

    def load_or_train(self):
        texts = load_all_corpus_files()
        self.ngram.train(texts)

        if TF_AVAILABLE and os.path.exists(LSTM_MODEL):
            try:
                self.lstm = load_model(LSTM_MODEL)
                with open(TOKENIZER_FILE, "rb") as f:
                    self.tokenizer = pickle.load(f)
                logger.info("✅ LSTM model loaded from file.")
            except Exception as e:
                logger.warning(
                    f"⚠️  Could not load {LSTM_MODEL} ({e}). "
                    "Falling back to N-gram only. Run train_lstm.py to rebuild."
                )
                self.lstm = None
        elif TF_AVAILABLE and texts:
            self.train_lstm(texts, epochs=20, verbose=0)

    def train_lstm(self, texts, epochs=50, batch_size=32, verbose=1):
        logger.info(f"🔁 Training LSTM (Epochs: {epochs})...")
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(texts)

        sequences = []
        for t in texts:
            seq = self.tokenizer.texts_to_sequences([t])[0]
            for i in range(1, len(seq)):
                sequences.append(seq[:i + 1])

        if not sequences:
            logger.warning("No sequences generated for training.")
            return

        maxlen = max(len(s) for s in sequences)
        # padding 'pre' is standard for LSTM to look at ending, but here we predict next word given history.
        # standard Keras pad_sequences defaults to 'pre'.
        sequences = pad_sequences(sequences, maxlen=maxlen)
        X, y = sequences[:, :-1], sequences[:, -1]

        vocab_size = len(self.tokenizer.word_index) + 1
        
        self.lstm = Sequential([
            Embedding(vocab_size, 64, input_length=X.shape[1]),
            LSTM(128),
            Dense(vocab_size, activation="softmax")
        ])

        self.lstm.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=['accuracy']
        )
        self.lstm.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose)

        self.lstm.save(LSTM_MODEL)
        with open(TOKENIZER_FILE, "wb") as f:
            pickle.dump(self.tokenizer, f)

        logger.info("✅ LSTM trained")

    def schedule_lstm_retrain(self, min_samples=20):
        if not TF_AVAILABLE or self.is_retraining or self.new_samples < min_samples:
            return

        def retrain():
            self.is_retraining = True
            try:
                texts = load_all_corpus_files()
                self.train_lstm(texts, epochs=10, verbose=0)
                self.new_samples = 0
            finally:
                self.is_retraining = False

        threading.Thread(target=retrain, daemon=True).start()

    def predict_next_word(self, context, top_k=TOP_K, use_lstm=False):
        context = clean_text(context)

        if use_lstm and self.lstm and not self.is_retraining:
            seq = self.tokenizer.texts_to_sequences([context])[0]
            if not seq:
                return []
            seq = pad_sequences([seq], maxlen=self.lstm.input_shape[1])
            preds = self.lstm.predict(seq, verbose=0)[0]
            idxs = preds.argsort()[-top_k:][::-1]
            rev = {v: k for k, v in self.tokenizer.word_index.items()}
            return [
                Prediction(rev.get(i, ""), float(preds[i]))
                for i in idxs
            ]

        return self.ngram.predict(context)

    def complete_word(self, partial, context="", top_k=TOP_K):
        return self.ngram.complete(partial)
