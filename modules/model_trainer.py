import logging
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import spacy
import json


nlp = spacy.load("en_core_web_sm")

# Ensure sentence boundaries can be detected
if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")

# Configure logger
logger = logging.getLogger(__name__)

# Download required NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    # If model not found, use a simpler approach
    logger.warning("SpaCy model not found. Using default English model.")
    nlp = spacy.blank('en')  # Use blank model as fallback

class EssayFeedbackModel:
    """Class for training and evaluating essay feedback models"""

    def __init__(self, model_dir='./models'):
        """
        Initialize the EssayFeedbackModel class

        Args:
            model_dir (str): Directory to save trained models
        """
        self.model_dir = model_dir
        self.vectorizer = None
        self.models = {
            'grammar': None,
            'structure': None,
            'coherence': None,
            'vocabulary': None,
            'overall': None
        }

        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    def preprocess_text(self, text):
        """
        Preprocess and tokenize essay text

        Args:
            text (str): The essay text

        Returns:
            str: Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()

        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)

        # Tokenize and remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words]

        # Join tokens back to text
        return ' '.join(tokens)

    def extract_features(self, texts, train=False):
        """
        Extract features from preprocessed essay texts

        Args:
            texts (list): List of preprocessed essay texts
            train (bool): Whether to fit the vectorizer or just transform

        Returns:
            numpy.ndarray: Feature matrix
        """
        if train or self.vectorizer is None:
            # Create and fit TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2
            )
            feature_matrix = self.vectorizer.fit_transform(texts)
        else:
            # Transform using existing vectorizer
            feature_matrix = self.vectorizer.transform(texts)

        return feature_matrix

    def extract_additional_features(self, texts):
        """
        Extract additional linguistic features from essays

        Args:
            texts (list): List of original essay texts

        Returns:
            numpy.ndarray: Additional feature matrix
        """
        features = []
        total = len(texts)

        print(f"\nProcessing {total} essays...")
        for i, text in enumerate(texts, 1):
            if i % 100 == 0:
                print(f"Progress: {i}/{total} essays processed")

            # Number of paragraphs
            paragraphs = text.split('\n\n')
            num_paragraphs = len(paragraphs)

            # Process with spaCy
            doc = nlp(text)

            # Average sentence length
            sentences = list(doc.sents)
            avg_sent_len = np.mean([len(sent) for sent in sentences]) if sentences else 0

            # Vocabulary richness (type-token ratio)
            tokens = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]
            unique_tokens = set(tokens)
            ttr = len(unique_tokens) / len(tokens) if tokens else 0

            # Number of discourse markers (connectives)
            discourse_markers = ['however', 'therefore', 'thus', 'furthermore', 
                              'moreover', 'in addition', 'consequently', 
                              'nevertheless', 'in contrast', 'on the other hand',
                              'for example', 'in conclusion']
            num_markers = sum(1 for marker in discourse_markers if marker in text.lower())

            # Part of speech counts
            pos_counts = {}
            for token in doc:
                pos = token.pos_
                pos_counts[pos] = pos_counts.get(pos, 0) + 1

            # Normalized POS counts (as percentage of total tokens)
            total_tokens = len(doc)
            noun_pct = pos_counts.get('NOUN', 0) / total_tokens if total_tokens > 0 else 0
            verb_pct = pos_counts.get('VERB', 0) / total_tokens if total_tokens > 0 else 0
            adj_pct = pos_counts.get('ADJ', 0) / total_tokens if total_tokens > 0 else 0
            adv_pct = pos_counts.get('ADV', 0) / total_tokens if total_tokens > 0 else 0

            # Combine features
            essay_features = [
                num_paragraphs,
                len(text),
                len(sentences),
                avg_sent_len,
                ttr,
                num_markers,
                noun_pct,
                verb_pct,
                adj_pct,
                adv_pct
            ]

            features.append(essay_features)

        return np.array(features)

    def load_and_preprocess_data(self, data_path):
        """
        Load and preprocess the essay-feedback dataset

        Args:
            data_path (str): Path to the dataset CSV file

        Returns:
            tuple: X (features), y (targets), X_raw (original texts)
        """
        # Load dataset
        df = pd.read_csv(data_path)

        # Extract features and targets
        essays = df['Essay'].tolist()
        scores = {
            'grammar': df['Grammar_Score'].tolist() if 'Grammar_Score' in df.columns else None,
            'structure': df['Structure_Score'].tolist() if 'Structure_Score' in df.columns else None,
            'coherence': df['Coherence_Score'].tolist() if 'Coherence_Score' in df.columns else None,
            'vocabulary': df['Vocabulary_Score'].tolist() if 'Vocabulary_Score' in df.columns else None,
            'overall': df['Score'].tolist() if 'Score' in df.columns else None
        }

        # Preprocess texts
        preprocessed_essays = [self.preprocess_text(essay) for essay in essays]

        # Extract features
        features_tfidf = self.extract_features(preprocessed_essays, train=True)
        features_linguistic = self.extract_additional_features(essays)

        # Convert targets to numpy arrays
        for category in scores:
            if scores[category] is not None:
                scores[category] = np.array(scores[category])

        return features_tfidf, features_linguistic, scores, essays

    def train_model(self, X_tfidf, X_linguistic, y, category):
        """
        Train a model for a specific feedback category

        Args:
            X_tfidf (scipy.sparse.csr_matrix): TF-IDF features
            X_linguistic (numpy.ndarray): Linguistic features
            y (numpy.ndarray): Target scores
            category (str): Feedback category

        Returns:
            object: Trained model
        """
        if y is None:
            logger.warning(f"No target data for category: {category}")
            return None

        logger.info(f"Training model for category: {category}")

        # Convert sparse matrix to dense for combining with linguistic features
        X_tfidf_dense = X_tfidf.toarray()

        # Combine features
        X_combined = np.hstack((X_tfidf_dense, X_linguistic))

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y, test_size=0.2, random_state=42
        )

        # Train model (Random Forest for regression)
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        logger.info(f"Model {category} RMSE: {rmse:.4f}")

        # Save model
        model_path = os.path.join(self.model_dir, f"{category}_model.joblib")
        joblib.dump(model, model_path)

        return model

    def train_all_models(self, data_path):
        """
        Train models for all feedback categories if they don't exist

        Args:
            data_path (str): Path to the dataset CSV file

        Returns:
            dict: Dictionary of trained models
        """
        # Check if all models exist
        all_models_exist = True
        for category in self.models:
            model_path = os.path.join(self.model_dir, f"{category}_model.joblib")
            if not os.path.exists(model_path):
                all_models_exist = False
                break

        vectorizer_path = os.path.join(self.model_dir, "tfidf_vectorizer.joblib")
        if not os.path.exists(vectorizer_path):
            all_models_exist = False

        # If all models exist, load them instead of training
        if all_models_exist:
            logger.info("Found existing models. Loading from disk...")
            return self.load_models()

        # If models don't exist, train new ones
        logger.info("Training new models...")
        X_tfidf, X_linguistic, scores, raw_essays = self.load_and_preprocess_data(data_path)

        # Train models for each category
        for category in self.models:
            if scores[category] is not None:
                self.models[category] = self.train_model(
                    X_tfidf, X_linguistic, scores[category], category
                )

        # Save vectorizer
        joblib.dump(self.vectorizer, vectorizer_path)
        logger.info("Models trained and saved successfully")

        return self.models

    def load_models(self):
        """
        Load trained models from disk

        Returns:
            dict: Dictionary of loaded models
        """
        # Load vectorizer
        vectorizer_path = os.path.join(self.model_dir, "tfidf_vectorizer.joblib")
        if os.path.exists(vectorizer_path):
            self.vectorizer = joblib.load(vectorizer_path)

        # Load models for each category
        for category in self.models:
            model_path = os.path.join(self.model_dir, f"{category}_model.joblib")
            if os.path.exists(model_path):
                self.models[category] = joblib.load(model_path)
                logger.info(f"Loaded model for category: {category}")

        return self.models

    def predict(self, essay_text):
        """
        Generate feedback for a new essay

        Args:
            essay_text (str): The essay text

        Returns:
            dict: Dictionary of feedback scores
        """
        # Check if models are loaded
        if all(model is None for model in self.models.values()):
            try:
                self.load_models()
            except Exception as e:
                logger.error(f"Error loading models: {str(e)}")
                return None

        # Preprocess text
        preprocessed_text = self.preprocess_text(essay_text)

        # Extract features
        X_tfidf = self.extract_features([preprocessed_text], train=False)
        X_linguistic = self.extract_additional_features([essay_text])

        # Convert sparse matrix to dense for combining with linguistic features
        X_tfidf_dense = X_tfidf.toarray()

        # Combine features
        X_combined = np.hstack((X_tfidf_dense, X_linguistic))

        # Generate predictions
        predictions = {}
        for category, model in self.models.items():
            if model is not None:
                score = model.predict(X_combined)[0]
                # Round to 2 decimal places and ensure in reasonable range
                predictions[category] = round(min(max(score, 0), 10), 2)

        return predictions

# Function to prepare IELTS dataset for model training
def prepare_ielts_dataset(input_path, output_path):
    """
    Prepare IELTS dataset for model training by extracting scores

    Args:
        input_path (str): Path to the raw IELTS dataset
        output_path (str): Path to save the prepared dataset

    Returns:
        str: Path to the prepared dataset
    """
    try:
        # Load the dataset
        df = pd.read_csv(input_path)

        # Create empty columns for component scores if they don't exist
        if 'Grammar_Score' not in df.columns:
            df['Grammar_Score'] = None
        if 'Structure_Score' not in df.columns:
            df['Structure_Score'] = None
        if 'Coherence_Score' not in df.columns:
            df['Coherence_Score'] = None
        if 'Vocabulary_Score' not in df.columns:
            df['Vocabulary_Score'] = None

        # For IELTS scores, derive component scores based on total score
        # This is a simplification - ideally we would have actual component scores
        if 'Score' in df.columns:
            for index, row in df.iterrows():
                overall_score = row['Score']

                # If we have overall score but no component scores, estimate them
                if not pd.isna(overall_score):
                    # Convert IELTS band score (0-9) to our scale (0-10)
                    normalized_score = (overall_score / 9) * 10

                    # Add random variation for different components (±1 point)
                    grammar_score = min(max(normalized_score + np.random.uniform(-1, 1), 0), 10)
                    structure_score = min(max(normalized_score + np.random.uniform(-1, 1), 0), 10)
                    coherence_score = min(max(normalized_score + np.random.uniform(-1, 1), 0), 10)
                    vocabulary_score = min(max(normalized_score + np.random.uniform(-1, 1), 0), 10)

                    # Assign component scores only if they were originally None
                    if pd.isna(df.at[index, 'Grammar_Score']):
                        df.at[index, 'Grammar_Score'] = grammar_score
                    if pd.isna(df.at[index, 'Structure_Score']):
                        df.at[index, 'Structure_Score'] = structure_score
                    if pd.isna(df.at[index, 'Coherence_Score']):
                        df.at[index, 'Coherence_Score'] = coherence_score
                    if pd.isna(df.at[index, 'Vocabulary_Score']):
                        df.at[index, 'Vocabulary_Score'] = vocabulary_score

        # Save the prepared dataset
        df.to_csv(output_path, index=False)

        logger.info(f"Prepared dataset saved to: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error preparing dataset: {str(e)}")
        raise

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Prepare dataset
    raw_data_path = "path/to/ielts_dataset.csv"
    prepared_data_path = "path/to/prepared_ielts_dataset.csv"
    prepare_ielts_dataset(raw_data_path, prepared_data_path)

    # Initialize and train models
    model_trainer = EssayFeedbackModel()
    models = model_trainer.train_all_models(prepared_data_path)

    # Test prediction
    test_essay = "This is a test essay for demonstration purposes."
    predictions = model_trainer.predict(test_essay)
    print("Predictions:", predictions)