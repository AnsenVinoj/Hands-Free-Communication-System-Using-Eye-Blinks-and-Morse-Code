import os
from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai
from dataclasses import dataclass
import next_word_predictor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini
# Try to get key from env, otherwise it might be set via other means or will fail gracefully later
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

MODEL_NAME = "models/gemini-1.5-flash" # Updated to a faster, valid model if 'gemini-3-pro' is invalid or nonexistent. 'gemini-1.5-flash' is good for latency.

@dataclass
class Prediction:
    word: str
    probability: float

    def to_dict(self):
        return {"word": self.word, "probability": self.probability}


class GeminiWordPredictor:
    def __init__(self, top_k=3):
        self.model = genai.GenerativeModel(MODEL_NAME)
        self.top_k = top_k
        # Initialize internal N-gram predictor for fallback and completion
        self.ngram = next_word_predictor.NGramPredictor()
        self.is_retraining = False # minimal state for compatibility

    def load_or_train(self):
        """Load corpus and train the internal N-gram model."""
        logger.info("Loading and training N-gram model...")
        texts = next_word_predictor.load_all_corpus_files()
        if not texts:
            logger.warning("No corpus files found in 'corpus/'. N-gram model will be empty.")
        self.ngram.train(texts)
        logger.info("N-gram model trained.")

    def predict_next_word(self, context: str):
        """
        Predict next word using Gemini, falling back to N-gram on error or empty context.
        """
        if not context.strip():
            return []

        try:
            prompt = f"""
You are an advanced predictive text engine assisting a person with limited mobility who uses eye-tracking to communicate.
Your goal is to predict the next single word the user wants to say, based on the context.
Prioritize:
1. Immediate needs (e.g., water, help, pain, doctor).
2. Common conversational flow (e.g., thank you, hello, how).
3. Contextual relevance to the previous words.

Context:
"{context}"

Task:
Predict the next {self.top_k} most likely single words.
Return ONLY a comma-separated list of 3-5 words in lowercase. Do not include punctuation.
"""
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            
            # Simple parsing
            words = [w.strip().lower() for w in text.split(",") if w.strip()]
            words = [w for w in words if " " not in w] # Ensure single words
            words = words[:self.top_k]

            if not words:
                raise ValueError("No words returned from Gemini")

            # formatting
            prob = round(1.0 / max(len(words), 1), 3)
            return [Prediction(w, prob) for w in words]

        except Exception as e:
            logger.error(f"Gemini error: {e}. Falling back to N-gram.")
            # Fallback to N-gram
            # We need to adapt N-gram output (which uses next_word_predictor.Prediction) to our Prediction class?
            # Actually they are identical dataclasses.
            return self.ngram.predict(context)

    def complete_word(self, partial: str, context: str = ""):
        """
        Complete the current word using N-gram model (faster/cheaper).
        """
        return self.ngram.complete(partial)

    def schedule_lstm_retrain(self):
        """
        Stub for compatibility. 
        """
        # If we aren't using LSTM in this hybrid class, we can just log or pass.
        # Or we could actually retrain the N-gram?
        # app.py calls learn_sentence then this.
        # ngram learn_sentence updates itself instantly.
        pass
