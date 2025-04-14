import os
import re
import email
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import VotingClassifier
import joblib
import hmmlearn.hmm as hmm
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SpamDetector:
    def __init__(self, model_path=None):
        """
        Initialize the spam detector with optional pre-trained models
        """
        self.naive_bayes_model = None
        self.svm_model = None
        self.hmm_spam_model = None
        self.hmm_ham_model = None
        self.vectorizer = None
        self.ensemble_model = None
        self.model_path = model_path if model_path else "models"

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
            logger.info(f"Created directory: {self.model_path}")

        # Try to load pre-trained models if they exist
        try:
            self.load_models()
            logger.info("Pre-trained models loaded successfully.")
        except Exception as e:
            logger.info(
                f"No pre-trained models found or error loading models: {str(e)}"
            )

    def preprocess_text(self, text):
        """
        Clean and preprocess the email text
        """
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove HTML tags
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text()

        # Remove email addresses
        text = re.sub(r"\S*@\S*\s?", "", text)

        # Remove URLs
        text = re.sub(r"http\S+", "", text)

        # Remove special characters and numbers
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\d+", "", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def extract_email_content(self, email_file):
        """
        Extract content from email file
        """
        try:
            with open(email_file, "r", encoding="utf-8", errors="ignore") as f:
                msg = email.message_from_file(f)

            # Check headers for spam indicators
            from_address = msg.get("from", "").lower()
            subject = msg.get("subject", "").lower()

            # Header spam indicators
            header_spam_score = 0
            suspicious_terms = [
                "spam",
                "scam",
                "spammer",
                "marketing",
                "offer",
                "prize",
                "winner",
            ]
            suspicious_domains = ["example.com", "temp.com", "tempmail", "freemail"]

            # Check From address
            if any(term in from_address for term in suspicious_terms):
                header_spam_score += 2.0
            if any(domain in from_address for domain in suspicious_domains):
                header_spam_score += 1.5

            # Check Subject
            if any(term in subject for term in suspicious_terms):
                header_spam_score += 1.5
            if subject.isupper() or "!!!" in subject:
                header_spam_score += 1.0

            # Get the body
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    if content_type == "text/plain" or content_type == "text/html":
                        try:
                            body += part.get_payload(decode=True).decode(
                                "utf-8", errors="ignore"
                            )
                        except:
                            body += str(part.get_payload())
            else:
                try:
                    body = msg.get_payload(decode=True).decode("utf-8", errors="ignore")
                except:
                    body = str(msg.get_payload())

            # Combine subject and body
            content = f"{subject} {body}"
            processed_content = self.preprocess_text(content)

            # Store header spam score in the content
            return processed_content, header_spam_score
        except Exception as e:
            logger.error(f"Error extracting content from {email_file}: {str(e)}")
            return "", 0.0

    def load_spamassassin_data(self, data_dir):
        """
        Load emails from the dataset
        """
        emails = []
        labels = []

        # Check if the dataset has the expected directory structure
        spam_dir = os.path.join(data_dir, "spam")
        ham_dir = os.path.join(data_dir, "ham")

        if not os.path.exists(spam_dir) or not os.path.exists(ham_dir):
            logger.error(
                f"Required directories not found. Need both {spam_dir} and {ham_dir}"
            )
            return emails, labels

        # Load ham (non-spam) emails
        for filename in os.listdir(ham_dir):
            if filename.startswith("."):
                continue
            filepath = os.path.join(ham_dir, filename)
            if os.path.isfile(filepath):
                content, _ = self.extract_email_content(filepath)
                if content:
                    emails.append(content)
                    labels.append(0)  # 0 for ham

        # Load spam emails
        for filename in os.listdir(spam_dir):
            if filename.startswith("."):
                continue
            filepath = os.path.join(spam_dir, filename)
            if os.path.isfile(filepath):
                content, _ = self.extract_email_content(filepath)
                if content:
                    emails.append(content)
                    labels.append(1)  # 1 for spam

        # Check if we have enough data
        if len(emails) == 0:
            logger.error(f"No emails found in {data_dir}")
            return emails, labels

        logger.info(
            f"Loaded {len(emails)} emails: {labels.count(0)} ham, {labels.count(1)} spam"
        )
        return emails, labels

    def train_hmm(self, texts, labels):
        """
        Train Hidden Markov Models for spam and ham
        """
        # Separate spam and ham
        spam_texts = [text for text, label in zip(texts, labels) if label == 1]
        ham_texts = [text for text, label in zip(texts, labels) if label == 0]

        # Convert texts to sequences for HMM
        # Using character bigrams as a simple sequence
        def text_to_sequence(text):
            return [ord(c) % 128 for c in text]  # Use ASCII values modulo 128

        spam_sequences = [
            text_to_sequence(text)[:100] for text in spam_texts
        ]  # Limit sequence length
        ham_sequences = [text_to_sequence(text)[:100] for text in ham_texts]

        # Pad sequences to make them uniform length
        max_len = 100
        spam_sequences = [
            seq + [0] * (max_len - len(seq))
            for seq in spam_sequences
            if len(seq) <= max_len
        ]
        ham_sequences = [
            seq + [0] * (max_len - len(seq))
            for seq in ham_sequences
            if len(seq) <= max_len
        ]

        # Convert to numpy arrays
        spam_sequences = np.array(spam_sequences)
        ham_sequences = np.array(ham_sequences)

        # Initialize and train HMM for spam
        self.hmm_spam_model = hmm.GaussianHMM(n_components=5, covariance_type="diag")
        try:
            self.hmm_spam_model.fit(np.vstack(spam_sequences))
        except Exception as e:
            logger.error(f"Error training spam HMM: {str(e)}")

        # Initialize and train HMM for ham
        self.hmm_ham_model = hmm.GaussianHMM(n_components=5, covariance_type="diag")
        try:
            self.hmm_ham_model.fit(np.vstack(ham_sequences))
        except Exception as e:
            logger.error(f"Error training ham HMM: {str(e)}")

    def hmm_predict(self, texts):
        """
        Predict using HMM models
        """
        predictions = []

        # Convert texts to sequences
        def text_to_sequence(text):
            return [ord(c) % 128 for c in text]  # Use ASCII values modulo 128

        for text in texts:
            seq = text_to_sequence(text)[:100]
            if len(seq) < 100:
                seq = seq + [0] * (100 - len(seq))

            seq = np.array([seq])

            try:
                spam_score = self.hmm_spam_model.score(seq)
                ham_score = self.hmm_ham_model.score(seq)

                # Predict the class with higher log probability
                if spam_score > ham_score:
                    predictions.append(1)  # Spam
                else:
                    predictions.append(0)  # Ham
            except Exception as e:
                logger.error(f"Error in HMM prediction: {str(e)}")
                predictions.append(0)  # Default to ham on error

        return np.array(predictions)

    def train(self, data_dir):
        """
        Train the spam detection models
        """
        # Load and preprocess the data
        emails, labels = self.load_spamassassin_data(data_dir)

        if not emails:
            logger.error("No email data found. Make sure the dataset path is correct.")
            return False

        if len(emails) < 10:  # Require at least 10 emails for training
            logger.error(
                f"Not enough emails found ({len(emails)}). Need at least 10 emails for training."
            )
            return False

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            emails, labels, test_size=0.2, random_state=42
        )

        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.7)

        # Transform the text data to TF-IDF features
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)

        # Train Naive Bayes model
        logger.info("Training Naive Bayes model...")
        self.naive_bayes_model = MultinomialNB()
        self.naive_bayes_model.fit(X_train_tfidf, y_train)

        # Train SVM model
        logger.info("Training SVM model...")
        self.svm_model = SVC(kernel="linear", probability=True)
        self.svm_model.fit(X_train_tfidf, y_train)

        # Create an ensemble model
        logger.info("Creating ensemble model...")
        self.ensemble_model = VotingClassifier(
            estimators=[
                ("naive_bayes", self.naive_bayes_model),
                ("svm", self.svm_model),
            ],
            voting="soft",
        )
        self.ensemble_model.fit(X_train_tfidf, y_train)

        # Evaluate the models
        logger.info("Evaluating models...")

        # Naive Bayes evaluation
        nb_predictions = self.naive_bayes_model.predict(X_test_tfidf)
        nb_accuracy = accuracy_score(y_test, nb_predictions)
        logger.info(f"Naive Bayes Accuracy: {nb_accuracy:.4f}")
        logger.info("\nNaive Bayes Classification Report:")
        logger.info(classification_report(y_test, nb_predictions))

        # SVM evaluation
        svm_predictions = self.svm_model.predict(X_test_tfidf)
        svm_accuracy = accuracy_score(y_test, svm_predictions)
        logger.info(f"SVM Accuracy: {svm_accuracy:.4f}")
        logger.info("\nSVM Classification Report:")
        logger.info(classification_report(y_test, svm_predictions))

        # Save the models
        self.save_models()

        return True

    def save_models(self):
        """
        Save trained models to disk
        """
        joblib.dump(self.vectorizer, os.path.join(self.model_path, "vectorizer.pkl"))
        joblib.dump(
            self.naive_bayes_model,
            os.path.join(self.model_path, "naive_bayes_model.pkl"),
        )
        joblib.dump(self.svm_model, os.path.join(self.model_path, "svm_model.pkl"))
        joblib.dump(
            self.ensemble_model, os.path.join(self.model_path, "ensemble_model.pkl")
        )
        joblib.dump(
            self.hmm_spam_model, os.path.join(self.model_path, "hmm_spam_model.pkl")
        )
        joblib.dump(
            self.hmm_ham_model, os.path.join(self.model_path, "hmm_ham_model.pkl")
        )
        logger.info("Models saved successfully.")

    def load_models(self):
        """
        Load trained models from disk
        """
        self.vectorizer = joblib.load(os.path.join(self.model_path, "vectorizer.pkl"))
        self.naive_bayes_model = joblib.load(
            os.path.join(self.model_path, "naive_bayes_model.pkl")
        )
        self.svm_model = joblib.load(os.path.join(self.model_path, "svm_model.pkl"))
        self.ensemble_model = joblib.load(
            os.path.join(self.model_path, "ensemble_model.pkl")
        )
        self.hmm_spam_model = joblib.load(
            os.path.join(self.model_path, "hmm_spam_model.pkl")
        )
        self.hmm_ham_model = joblib.load(
            os.path.join(self.model_path, "hmm_ham_model.pkl")
        )

    def predict_email(self, email_content, header_spam_score=0.0, method="ensemble"):
        """
        Predict if an email is spam or ham

        Args:
            email_content (str): The content of the email
            header_spam_score (float): Spam score from email headers
            method (str): The prediction method ('nb', 'svm', or 'ensemble')

        Returns:
            tuple: (prediction (1 for spam, 0 for ham), probability)
        """
        if not self.ensemble_model:
            logger.error("Models not trained. Please train the models first.")
            return None, None

        # Preprocess the email content
        processed_content = self.preprocess_text(email_content)

        # Check for very suspicious email addresses
        if isinstance(email_content, str) and "@" in email_content:
            suspicious_domains = [
                "example.com",
                "temp.com",
                "tempmail",
                "freemail",
                "spam",
                "scam",
            ]
            if any(domain in email_content.lower() for domain in suspicious_domains):
                header_spam_score += 2.0

        # Rest of the spam indicators remain the same
        spam_indicators = {
            "urgent": 1.5,
            "winner": 1.5,
            "won": 1.0,
            "prize": 1.5,
            "lottery": 2.0,
            "million": 1.5,
            "dollars": 1.0,
            "credit card": 1.5,
            "bank account": 1.5,
            "nigeria": 2.0,
            "inheritance": 1.5,
            "prince": 1.5,
            "congratulations": 1.0,
            "click here": 1.5,
            "act now": 1.5,
            "limited time": 1.0,
            "offer": 0.5,
            "free": 1.0,
            "discount": 0.5,
            "save": 0.5,
            "money": 0.5,
            "cash": 1.0,
            "investment": 1.0,
            "opportunity": 0.5,
            "guaranteed": 1.5,
            "no risk": 1.5,
            "viagra": 2.5,
            "medicine": 1.0,
            "pharmacy": 1.0,
            "prescription": 1.0,
            "weight loss": 1.5,
            "work from home": 1.5,
            "make money": 1.5,
            "earn": 1.0,
            "income": 0.5,
            "rich": 1.0,
            "wealthy": 1.0,
            "success": 0.5,
            "debt": 1.0,
            "loan": 1.0,
            "mortgage": 1.0,
            "refinance": 1.0,
            "insurance": 0.5,
            "spam": 2.5,
            "scam": 2.5,
            "virus": 1.5,
            "hack": 1.5,
            "account suspended": 1.5,
            "verify your account": 1.5,
            "security alert": 1.5,
            "unusual activity": 1.5,
            "password expired": 1.5,
            "win": 1.5,
            "selected": 1.0,
            "congratulation": 1.0,
            "100% free": 1.5,
            "best price": 1.0,
            "buy now": 1.0,
            "order now": 1.0,
            "special offer": 1.0,
            "great deal": 1.0,
        }

        # Calculate weighted spam score
        content_lower = processed_content.lower()
        spam_score = 0
        total_weight = 0
        for indicator, weight in spam_indicators.items():
            if indicator in content_lower:
                spam_score += weight
                total_weight += weight

        # Normalize spam score (0 to 1) and add header score with reduced weight
        indicator_score = (
            spam_score / (total_weight + 1) if total_weight > 0 else 0
        ) + (header_spam_score / 8.0)

        # Vectorize the content
        content_tfidf = self.vectorizer.transform([processed_content])

        # Get predictions from each model
        nb_pred = self.naive_bayes_model.predict(content_tfidf)[0]
        nb_prob = self.naive_bayes_model.predict_proba(content_tfidf)[0][1]

        svm_pred = self.svm_model.predict(content_tfidf)[0]
        svm_prob = self.svm_model.predict_proba(content_tfidf)[0][1]

        # Combine probabilities with balanced weights
        ensemble_prob = (2 * nb_prob + 2 * svm_prob + 1.5 * indicator_score) / 5.5

        # Calculate ensemble probability
        ensemble_prob = (
            ensemble_prob + 1.5 * indicator_score + header_spam_score / 8.0
        ) / 3.5

        # Determine if email is spam based on multiple criteria
        is_spam = (
            ensemble_prob > 0.47  # Slightly higher ensemble threshold
            and (
                indicator_score > 0.6 or header_spam_score > 2.5
            )  # Stricter indicator requirements
        ) or header_spam_score > 4.0  # Keep high header score override

        return 1 if is_spam else 0, ensemble_prob

    def process_email_file(self, file_path):
        """
        Process a single email file and detect if it's spam.
        """
        # Extract email content and header score
        content, header_spam_score = self.extract_email_content(file_path)

        if not content:
            logger.warning(f"Could not extract content from {file_path}")
            return None

        # Process with spam detector
        prediction, probability = self.predict_email(content, header_spam_score)

        result = {
            "is_spam": bool(prediction),
            "spam_probability": float(probability),
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "processed_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        return result


def process_email(email_content, spam_detector):
    """
    Process an email and determine if it's spam

    Args:
        email_content (str): The content of the email
        spam_detector (SpamDetector): Trained spam detector instance

    Returns:
        dict: Prediction results
    """
    # If email_content is a tuple (from extract_email_content), use both values
    if isinstance(email_content, tuple):
        content, header_spam_score = email_content
    else:
        content, header_spam_score = email_content, 0.0

    prediction, probability = spam_detector.predict_email(content, header_spam_score)

    result = {
        "is_spam": bool(prediction),
        "spam_probability": float(probability),
        "prediction_method": "ensemble",
    }

    # Get individual model predictions
    nb_pred, nb_prob = spam_detector.predict_email(
        content, header_spam_score, method="nb"
    )
    svm_pred, svm_prob = spam_detector.predict_email(
        content, header_spam_score, method="svm"
    )

    result["model_predictions"] = {
        "naive_bayes": {"is_spam": bool(nb_pred), "probability": float(nb_prob)},
        "svm": {"is_spam": bool(svm_pred), "probability": float(svm_prob)},
    }

    return result


if __name__ == "__main__":
    # Example usage
    spam_detector = SpamDetector()

    # Path to email dataset
    data_dir = os.path.join(
        os.path.dirname(__file__), "processed_emails"
    )  # Directory containing spam and ham folders

    # Train if models don't exist
    if not os.path.exists(os.path.join(spam_detector.model_path, "ensemble_model.pkl")):
        print("Training models...")
        spam_detector.train(data_dir)
    else:
        print("Pre-trained models found. Do you want to retrain? (y/n)")
        choice = input().lower()
        if choice == "y":
            print("Training new models...")
            spam_detector.train(data_dir)
        else:
            print("Loading pre-trained models...")
            try:
                spam_detector.load_models()
            except Exception as e:
                print(f"Error loading models: {str(e)}")
                print("Training new models...")
                spam_detector.train(data_dir)

    # Example email
    test_email = """
    Subject: FREE offer - act NOW!!!
    
    Dear Friend,
    
    Congratulations! You've been selected to receive our exclusive offer.
    Make $5000 a week working from home! This is NOT a scam.
    
    Click here: http://totally-not-a-scam.com
    
    Limited time offer! ACT NOW!!!
    """

    # Predict
    result = process_email(test_email, spam_detector)

    print("\nPrediction Results:")
    print(f"Is Spam: {result['is_spam']}")
    print(f"Spam Probability: {result['spam_probability']:.4f}")
    print("\nIndividual Model Predictions:")
    for model, pred in result["model_predictions"].items():
        print(
            f"  {model}: Is Spam = {pred['is_spam']}, Probability = {pred['probability']:.4f}"
        )
