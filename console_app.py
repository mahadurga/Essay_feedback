import nltk
import spacy
from nltk.tokenize import sent_tokenize
import sys
import os

# Add the parent directory to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.model_trainer import EssayFeedbackModel

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')

from modules.dataset_loader import download_ielts_dataset, load_ielts_dataset

# Download and prepare dataset
dataset_path = download_ielts_dataset()
df = load_ielts_dataset(dataset_path)

# Ensure directory exists
os.makedirs("EssayFeedbackAI", exist_ok=True)
df.to_csv("EssayFeedbackAI/ielts_writing_dataset.csv", index=False)

import threading
import time

print("\nInitializing model...")
model = EssayFeedbackModel()

# Check if models directory exists and has all required model files
model_files = [
    "grammar_model.joblib",
    "structure_model.joblib", 
    "coherence_model.joblib",
    "vocabulary_model.joblib",
    "overall_model.joblib",
    "tfidf_vectorizer.joblib"
]

models_exist = all(os.path.exists(os.path.join("EssayFeedbackAI/models", f)) for f in model_files)

if models_exist:
    print("Found existing trained models, loading them...")
    model.load_models()
    print("Models loaded successfully!")
else:
    print("Training models for the first time...")
    
    def train_model_thread():
        try:
            model.train_all_models("EssayFeedbackAI/ielts_writing_dataset.csv")
            print("\nModel training completed successfully!")
        except Exception as e:
            print(f"\nError during training: {str(e)}")

# Start training in a separate thread
training_thread = threading.Thread(target=train_model_thread)
training_thread.daemon = True  # Allow the thread to be terminated when main program exits
training_thread.start()

try:
    print("Training models in background (this may take a few minutes)...")
    while training_thread.is_alive():
        time.sleep(1)
except KeyboardInterrupt:
    print("\nTraining interrupted. Loading pre-trained models if available...")
    model.load_models()
except Exception as e:
    print(f"\nError during training: {str(e)}")
    print("Attempting to load pre-trained models...")
    model.load_models()

def get_menu_choice():
    print("\n=== Essay Feedback System ===")
    print("1. Submit new essay")
    print("2. Upload essay file")
    print("3. Exit")
    return input("Choose an option (1-3): ")

def get_essay():
    print("\nEnter your essay (type 'END' on a new line when finished):")
    lines = []
    while True:
        line = input()
        if line.strip().upper() == 'END':
            break
        lines.append(line)
    return '\n'.join(lines)

def get_essay_from_file():
    filename = input("\nEnter the path to your essay file: ")
    try:
        with open(filename, 'r') as f:
            return f.read()
    except:
        print("Error reading file. Please check the path and try again.")
        return None

def analyze_essay(text):
    # Basic analysis
    sentences = sent_tokenize(text)
    words = text.split()

    # Basic metrics
    num_sentences = len(sentences)
    num_words = len(words)
    avg_sentence_length = num_words / num_sentences if num_sentences > 0 else 0

    # Get model predictions
    predictions = model.predict(text)
    score = predictions.get('overall', 5.0)  # Default to 5.0 if prediction fails

    return {
        'num_sentences': num_sentences,
        'num_words': num_words,
        'avg_sentence_length': avg_sentence_length,
        'score': score
    }

def display_feedback(analysis):
    print("\n=== Essay Analysis Results ===")
    print(f"\nScore: {analysis['score']:.1f} / 9.0")
    print(f"\nBasic Statistics:")
    print(f"- Number of sentences: {analysis['num_sentences']}")
    print(f"- Number of words: {analysis['num_words']}")
    print(f"- Average sentence length: {analysis['avg_sentence_length']:.1f} words")

def main():
    print("Welcome to the Essay Feedback System!")

    while True:
        choice = get_menu_choice()

        if choice == '1':
            essay = get_essay()
            if essay.strip():
                print("\nAnalyzing essay...")
                analysis = analyze_essay(essay)
                display_feedback(analysis)
            else:
                print("Essay cannot be empty!")

        elif choice == '2':
            essay = get_essay_from_file()
            if essay and essay.strip():
                print("\nAnalyzing essay...")
                analysis = analyze_essay(essay)
                display_feedback(analysis)

        elif choice == '3':
            print("\nThank you for using the Essay Feedback System!")
            break
        else:
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    main()