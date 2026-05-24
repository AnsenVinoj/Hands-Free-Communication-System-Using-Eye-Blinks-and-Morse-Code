"""
LSTM Model Training Script
==========================

Train the LSTM neural network model using the `WordPredictor` class.

Usage:
    python train_lstm.py
    
    # Custom epochs:
    python train_lstm.py --epochs 100

Author: Next Word Prediction Project
Version: 2.1.0
"""

import sys
import argparse
import time
from next_word_predictor import WordPredictor, load_all_corpus_files, TF_AVAILABLE

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train LSTM model on available corpus'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    print()
    print("=" * 60)
    print("🧠 LSTM Model Training")
    print("=" * 60)
    
    if not TF_AVAILABLE:
        print("\n❌ ERROR: TensorFlow/Keras is not installed or not working.")
        print("Please install tensorflow: pip install tensorflow")
        return 1

    print("\nLoading corpus...")
    texts = load_all_corpus_files()
    
    if not texts:
        print("❌ ERROR: No texts found in 'corpus' directory.")
        return 1
    
    print(f"📝 Found {len(texts)} sentences/lines.")
    
    predictor = WordPredictor()
    
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    start_time = time.time()
    
    try:
        # Train explicitly using the new public method
        predictor.train_lstm(texts, epochs=args.epochs, verbose=1)
        
        duration = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"✅ Training complete in {duration:.1f} seconds")
        print("=" * 60)
        
        # Test
        print("\n🧪 Testing predictions:")
        test_phrases = ["Pizza is", "I like", "The"]
        
        for phrase in test_phrases:
            preds = predictor.predict_next_word(phrase, top_k=3, use_lstm=True)
            print(f"\nInput: '{phrase}'")
            if not preds:
                print("   (No predictions)")
            for p in preds:
                print(f"   - {p.word} ({p.probability:.2%})")

        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted.")
        return 1
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
