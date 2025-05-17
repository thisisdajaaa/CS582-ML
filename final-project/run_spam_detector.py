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
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    auc,
)
from sklearn.ensemble import VotingClassifier
import joblib
import hmmlearn.hmm as hmm
import logging
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

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
        self.model_path = model_path if model_path else "models"
        self.training_history = {}
        self.visualization_dir = os.path.join(self.model_path, "visualizations")

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
            logger.info(f"Created directory: {self.model_path}")

        if not os.path.exists(self.visualization_dir):
            os.makedirs(self.visualization_dir)
            logger.info(f"Created directory: {self.visualization_dir}")

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

    def train_hmm(self, texts, labels, test_texts=None, test_labels=None):
        """
        Train Hidden Markov Models for spam and ham with enhanced visualization
        """
        # Separate spam and ham
        spam_texts = [text for text, label in zip(texts, labels) if label == 1]
        ham_texts = [text for text, label in zip(texts, labels) if label == 0]

        logger.info(
            f"Training HMM on {len(spam_texts)} spam and {len(ham_texts)} ham emails..."
        )

        # Convert texts to sequences for HMM
        # Using character bigrams as a simple sequence
        def text_to_sequence(text):
            return [ord(c) % 128 for c in text]  # Use ASCII values modulo 128

        # Process spam sequences with progress bar
        spam_sequences = []
        for text in tqdm(spam_texts, desc="Processing spam for HMM"):
            seq = text_to_sequence(text)[:100]  # Limit sequence length
            if len(seq) <= 100:  # Only use sequences that are short enough
                # Pad sequences to make them uniform length
                spam_sequences.append(seq + [0] * (100 - len(seq)))

        # Process ham sequences with progress bar
        ham_sequences = []
        for text in tqdm(ham_texts, desc="Processing ham for HMM"):
            seq = text_to_sequence(text)[:100]  # Limit sequence length
            if len(seq) <= 100:  # Only use sequences that are short enough
                # Pad sequences to make them uniform length
                ham_sequences.append(seq + [0] * (100 - len(seq)))

        # Convert to numpy arrays
        spam_sequences = np.array(spam_sequences)
        ham_sequences = np.array(ham_sequences)

        # Initialize and train HMM for spam
        logger.info("Training HMM for spam...")
        self.hmm_spam_model = hmm.GaussianHMM(
            n_components=5, covariance_type="diag", n_iter=20
        )
        try:
            # Track log likelihood during training
            self.training_history["hmm_spam_log_likelihood"] = []
            for i in range(5):  # Run multiple iterations, tracking convergence
                self.hmm_spam_model.fit(spam_sequences.reshape(-1, 1))
                self.training_history["hmm_spam_log_likelihood"].append(
                    self.hmm_spam_model.score(spam_sequences.reshape(-1, 1))
                )
                logger.info(
                    f"Spam HMM iteration {i+1}, log likelihood: {self.training_history['hmm_spam_log_likelihood'][-1]:.4f}"
                )
        except Exception as e:
            logger.error(f"Error training spam HMM: {str(e)}")

        # Initialize and train HMM for ham
        logger.info("Training HMM for ham...")
        self.hmm_ham_model = hmm.GaussianHMM(
            n_components=5, covariance_type="diag", n_iter=20
        )
        try:
            # Track log likelihood during training
            self.training_history["hmm_ham_log_likelihood"] = []
            for i in range(5):  # Run multiple iterations, tracking convergence
                self.hmm_ham_model.fit(ham_sequences.reshape(-1, 1))
                self.training_history["hmm_ham_log_likelihood"].append(
                    self.hmm_ham_model.score(ham_sequences.reshape(-1, 1))
                )
                logger.info(
                    f"Ham HMM iteration {i+1}, log likelihood: {self.training_history['hmm_ham_log_likelihood'][-1]:.4f}"
                )
        except Exception as e:
            logger.error(f"Error training ham HMM: {str(e)}")

        # Plot HMM training convergence
        self._plot_hmm_convergence()

        # If we have test data, evaluate HMM performance
        if test_texts is not None and test_labels is not None:
            predictions, probabilities = self.hmm_predict(test_texts)
            accuracy = accuracy_score(test_labels, predictions)
            logger.info(f"HMM Test Accuracy: {accuracy:.4f}")

            # Plot ROC curve for HMM
            self._plot_hmm_roc_curve(test_labels, probabilities)

    def _plot_hmm_convergence(self):
        """Plot the convergence of HMM training"""
        if (
            "hmm_spam_log_likelihood" in self.training_history
            and "hmm_ham_log_likelihood" in self.training_history
        ):
            plt.figure(figsize=(10, 6))
            plt.title("HMM Training Convergence")
            plt.plot(
                self.training_history["hmm_spam_log_likelihood"],
                "o-",
                color="r",
                label="Spam HMM",
            )
            plt.plot(
                self.training_history["hmm_ham_log_likelihood"],
                "o-",
                color="g",
                label="Ham HMM",
            )
            plt.xlabel("Iteration")
            plt.ylabel("Log Likelihood")
            plt.legend()
            plt.grid(True)

            plt.savefig(os.path.join(self.visualization_dir, "hmm_convergence.png"))
            plt.close()
            logger.info(
                f"HMM convergence plot saved to {self.visualization_dir}/hmm_convergence.png"
            )

    def _plot_hmm_roc_curve(self, test_labels, probabilities):
        """Plot ROC curve for HMM predictions"""
        plt.figure(figsize=(8, 6))

        # Calculate and plot ROC curve
        fpr, tpr, _ = roc_curve(test_labels, probabilities)
        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr, tpr, color="purple", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("HMM ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        plt.savefig(os.path.join(self.visualization_dir, "hmm_roc_curve.png"))
        plt.close()
        logger.info(
            f"HMM ROC curve saved to {self.visualization_dir}/hmm_roc_curve.png"
        )

    def _plot_model_comparison(
        self,
        y_test,
        nb_probabilities,
        svm_probabilities,
        hmm_probabilities,
    ):
        """Plot ROC curves for all models for comparison"""
        plt.figure(figsize=(10, 8))

        # Calculate ROC curve for Naive Bayes
        fpr_nb, tpr_nb, _ = roc_curve(y_test, nb_probabilities)
        roc_auc_nb = auc(fpr_nb, tpr_nb)
        plt.plot(
            fpr_nb,
            tpr_nb,
            color="blue",
            lw=2,
            label=f"Naive Bayes (AUC = {roc_auc_nb:.2f})",
        )

        # Calculate ROC curve for SVM
        fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_probabilities)
        roc_auc_svm = auc(fpr_svm, tpr_svm)
        plt.plot(
            fpr_svm, tpr_svm, color="red", lw=2, label=f"SVM (AUC = {roc_auc_svm:.2f})"
        )

        # Calculate ROC curve for HMM
        fpr_hmm, tpr_hmm, _ = roc_curve(y_test, hmm_probabilities)
        roc_auc_hmm = auc(fpr_hmm, tpr_hmm)
        plt.plot(
            fpr_hmm,
            tpr_hmm,
            color="purple",
            lw=2,
            label=f"HMM (AUC = {roc_auc_hmm:.2f})",
        )

        # Reference line for random classifier
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve Comparison")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        plt.savefig(os.path.join(self.visualization_dir, "model_comparison_roc.png"))
        plt.close()
        logger.info(
            f"Model comparison ROC curves saved to {self.visualization_dir}/model_comparison_roc.png"
        )

        # Add a second visualization showing individual model predictions vs. ensemble
        plt.figure(figsize=(12, 10))
        plt.scatter(
            nb_probabilities,
            svm_probabilities,
            c=hmm_probabilities,
            cmap="viridis",
            alpha=0.7,
            s=50,
            label="Points (color = HMM prob)",
        )

        # Add colorbar for HMM probabilities
        cbar = plt.colorbar()
        cbar.set_label("HMM Probability")

        # Add decision threshold line
        plt.axhline(y=0.5, color="r", linestyle="--", alpha=0.5)
        plt.axvline(x=0.5, color="r", linestyle="--", alpha=0.5)

        plt.xlabel("Naive Bayes Probability")
        plt.ylabel("SVM Probability")
        plt.title("Model Probability Distribution (Color = HMM Probability)")
        plt.grid(True, alpha=0.3)

        plt.savefig(
            os.path.join(self.visualization_dir, "probability_distribution_3d.png")
        )
        plt.close()
        logger.info(
            f"3D probability distribution saved to {self.visualization_dir}/probability_distribution_3d.png"
        )

    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all models and generate visualizations

        Args:
            X_test: Test features (preprocessed email content)
            y_test: Test labels (1 for spam, 0 for ham)
        """
        logger.info("Evaluating all models...")

        # Vectorize test data for ML models
        X_test_tfidf = self.vectorizer.transform(X_test)

        # Get predictions and probabilities from each model
        nb_predictions = self.naive_bayes_model.predict(X_test_tfidf)
        nb_probabilities = self.naive_bayes_model.predict_proba(X_test_tfidf)[:, 1]

        svm_predictions = self.svm_model.predict(X_test_tfidf)
        svm_probabilities = self.svm_model.predict_proba(X_test_tfidf)[:, 1]

        # Get HMM predictions
        hmm_predictions, hmm_probabilities = self.hmm_predict(X_test)

        # Calculate metrics for each model
        nb_accuracy = accuracy_score(y_test, nb_predictions)
        svm_accuracy = accuracy_score(y_test, svm_predictions)
        hmm_accuracy = accuracy_score(y_test, hmm_predictions)

        logger.info(f"Naive Bayes Accuracy: {nb_accuracy:.4f}")
        logger.info(f"SVM Accuracy: {svm_accuracy:.4f}")
        logger.info(f"HMM Accuracy: {hmm_accuracy:.4f}")

        # Print classification reports
        logger.info("\nNaive Bayes Classification Report:")
        logger.info(classification_report(y_test, nb_predictions))

        logger.info("\nSVM Classification Report:")
        logger.info(classification_report(y_test, svm_predictions))

        logger.info("\nHMM Classification Report:")
        logger.info(classification_report(y_test, hmm_predictions))

        # Generate ROC curves and other visualizations
        self._plot_model_comparison(
            y_test,
            nb_probabilities,
            svm_probabilities,
            hmm_probabilities,
        )

        # Plot confusion matrices
        self._plot_confusion_matrices(
            y_test,
            nb_predictions,
            svm_predictions,
            hmm_predictions,
        )

        # Store evaluation metrics in training history
        self.training_history["nb_accuracy"] = nb_accuracy
        self.training_history["svm_accuracy"] = svm_accuracy
        self.training_history["hmm_accuracy"] = hmm_accuracy

        return {
            "nb_accuracy": nb_accuracy,
            "svm_accuracy": svm_accuracy,
            "hmm_accuracy": hmm_accuracy,
        }

    # New function to plot confusion matrices
    def _plot_confusion_matrices(
        self,
        y_test,
        nb_predictions,
        svm_predictions,
        hmm_predictions,
    ):
        """Plot confusion matrices for all models"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Naive Bayes confusion matrix
        cm_nb = confusion_matrix(y_test, nb_predictions)
        sns.heatmap(cm_nb, annot=True, fmt="d", cmap="Blues", ax=axes[0, 0])
        axes[0, 0].set_title("Naive Bayes")
        axes[0, 0].set_xlabel("Predicted")
        axes[0, 0].set_ylabel("True")

        # SVM confusion matrix
        cm_svm = confusion_matrix(y_test, svm_predictions)
        sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Reds", ax=axes[0, 1])
        axes[0, 1].set_title("SVM")
        axes[0, 1].set_xlabel("Predicted")
        axes[0, 1].set_ylabel("True")

        # HMM confusion matrix
        cm_hmm = confusion_matrix(y_test, hmm_predictions)
        sns.heatmap(cm_hmm, annot=True, fmt="d", cmap="Purples", ax=axes[1, 0])
        axes[1, 0].set_title("HMM")
        axes[1, 0].set_xlabel("Predicted")
        axes[1, 0].set_ylabel("True")

        plt.tight_layout()
        plt.savefig(os.path.join(self.visualization_dir, "confusion_matrices.png"))
        plt.close()
        logger.info(
            f"Confusion matrices saved to {self.visualization_dir}/confusion_matrices.png"
        )

    def hmm_predict(self, texts):
        """
        Predict using HMM models
        """
        predictions = []
        probabilities = []

        # Convert texts to sequences
        def text_to_sequence(text):
            return [ord(c) % 128 for c in text]  # Use ASCII values modulo 128

        for text in texts:
            seq = text_to_sequence(text)[:100]
            if len(seq) < 100:
                seq = seq + [0] * (100 - len(seq))

            seq = np.array([seq])

            try:
                if self.hmm_spam_model is None or self.hmm_ham_model is None:
                    logger.warning(
                        "HMM models not trained. Defaulting to ham prediction."
                    )
                    predictions.append(0)
                    probabilities.append(0.0)
                    continue

                spam_score = self.hmm_spam_model.score(seq)
                ham_score = self.hmm_ham_model.score(seq)

                # Numerically stable softmax
                max_score = max(spam_score, ham_score)
                exp_spam = np.exp(spam_score - max_score)
                exp_ham = np.exp(ham_score - max_score)
                total_score = exp_spam + exp_ham

                # Handle potential division by zero or NaN
                if not np.isfinite(total_score) or total_score == 0.0:
                    spam_prob = 0.0
                else:
                    spam_prob = exp_spam / total_score
                    # If spam_prob is not finite, set to 0.0
                    if not np.isfinite(spam_prob):
                        spam_prob = 0.0

                # Predict the class with higher log probability
                if spam_score > ham_score:
                    predictions.append(1)  # Spam
                else:
                    predictions.append(0)  # Ham

                probabilities.append(spam_prob)

            except Exception as e:
                logger.error(f"Error in HMM prediction: {str(e)}")
                predictions.append(0)  # Default to ham on error
                probabilities.append(0.0)

        return np.array(predictions), np.array(probabilities)

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

        # Track Naive Bayes training metrics over time
        nb_train_scores = []
        nb_test_scores = []
        train_sizes = np.linspace(0.2, 1.0, 5)

        for size in tqdm(train_sizes, desc="NB Learning Curve"):
            # Calculate sample size
            train_size = int(len(X_train) * size)

            # Train on subset
            self.naive_bayes_model.fit(X_train_tfidf[:train_size], y_train[:train_size])

            # Evaluate
            train_score = self.naive_bayes_model.score(
                X_train_tfidf[:train_size], y_train[:train_size]
            )
            test_score = self.naive_bayes_model.score(X_test_tfidf, y_test)

            nb_train_scores.append(train_score)
            nb_test_scores.append(test_score)

        # Re-train on full dataset
        self.naive_bayes_model.fit(X_train_tfidf, y_train)

        # Store learning curve data
        self.training_history["nb_train_sizes"] = train_sizes
        self.training_history["nb_train_scores"] = nb_train_scores
        self.training_history["nb_test_scores"] = nb_test_scores

        # Plot Naive Bayes learning curve
        plt.figure(figsize=(10, 6))
        plt.title("Naive Bayes Learning Curve")
        plt.xlabel("Training Data Fraction")
        plt.ylabel("Accuracy Score")
        plt.grid()

        plt.plot(
            train_sizes, nb_train_scores, "o-", color="blue", label="Training Score"
        )
        plt.plot(train_sizes, nb_test_scores, "o-", color="orange", label="Test Score")

        plt.legend(loc="best")
        plt.savefig(
            os.path.join(self.visualization_dir, "naive_bayes_learning_curve.png")
        )
        plt.close()

        # Track final training metrics for Naive Bayes
        train_acc_nb = self.naive_bayes_model.score(X_train_tfidf, y_train)
        test_acc_nb = self.naive_bayes_model.score(X_test_tfidf, y_test)
        self.training_history["nb_train_accuracy"] = train_acc_nb
        self.training_history["nb_test_accuracy"] = test_acc_nb
        logger.info(
            f"Naive Bayes - Train Accuracy: {train_acc_nb:.4f}, Test Accuracy: {test_acc_nb:.4f}"
        )

        # Train SVM model
        logger.info("Training SVM model...")
        self.svm_model = SVC(kernel="linear", probability=True)

        # Track SVM training metrics over time
        svm_train_scores = []
        svm_test_scores = []

        for size in tqdm(train_sizes, desc="SVM Learning Curve"):
            # Calculate sample size
            train_size = int(len(X_train) * size)

            # Train on subset
            self.svm_model.fit(X_train_tfidf[:train_size], y_train[:train_size])

            # Evaluate
            train_score = self.svm_model.score(
                X_train_tfidf[:train_size], y_train[:train_size]
            )
            test_score = self.svm_model.score(X_test_tfidf, y_test)

            svm_train_scores.append(train_score)
            svm_test_scores.append(test_score)

        # Re-train on full dataset
        self.svm_model.fit(X_train_tfidf, y_train)

        # Store learning curve data
        self.training_history["svm_train_sizes"] = train_sizes
        self.training_history["svm_train_scores"] = svm_train_scores
        self.training_history["svm_test_scores"] = svm_test_scores

        # Plot SVM learning curve
        plt.figure(figsize=(10, 6))
        plt.title("SVM Learning Curve")
        plt.xlabel("Training Data Fraction")
        plt.ylabel("Accuracy Score")
        plt.grid()

        plt.plot(
            train_sizes, svm_train_scores, "o-", color="red", label="Training Score"
        )
        plt.plot(train_sizes, svm_test_scores, "o-", color="orange", label="Test Score")

        plt.legend(loc="best")
        plt.savefig(os.path.join(self.visualization_dir, "svm_learning_curve.png"))
        plt.close()

        # Track training metrics for SVM
        train_acc_svm = self.svm_model.score(X_train_tfidf, y_train)
        test_acc_svm = self.svm_model.score(X_test_tfidf, y_test)
        self.training_history["svm_train_accuracy"] = train_acc_svm
        self.training_history["svm_test_accuracy"] = test_acc_svm
        logger.info(
            f"SVM - Train Accuracy: {train_acc_svm:.4f}, Test Accuracy: {test_acc_svm:.4f}"
        )

        # Train HMM models
        logger.info("Training HMM models...")
        self.train_hmm(X_train, y_train, X_test, y_test)

        # Get HMM predictions for test data to include in evaluation
        hmm_predictions, hmm_probabilities = self.hmm_predict(X_test)

        # Evaluate all models
        self.evaluate_models(X_test, y_test)

        # Compare model performance with a bar chart
        self._plot_model_accuracy_comparison()

        # Save the models and training history
        self.save_models()

        return True

    def _plot_model_accuracy_comparison(self):
        """Create a bar chart comparing the accuracy of all models"""

        # Get accuracy values
        accuracies = [
            self.training_history.get("nb_test_accuracy", 0),
            self.training_history.get("svm_test_accuracy", 0),
            self.training_history.get("hmm_accuracy", 0),
        ]

        model_names = ["Naive Bayes", "SVM", "HMM"]

        # Create bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(model_names, accuracies, color=["blue", "red", "purple"])
        plt.title("Model Accuracy Comparison")
        plt.ylabel("Test Accuracy")
        plt.ylim(0, 1.0)

        # Add value labels on top of each bar
        for i, v in enumerate(accuracies):
            plt.text(i, v + 0.01, f"{v:.4f}", ha="center")

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.visualization_dir, "model_accuracy_comparison.png")
        )
        plt.close()

        logger.info(
            f"Model accuracy comparison saved to {self.visualization_dir}/model_accuracy_comparison.png"
        )

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
        self.hmm_spam_model = joblib.load(
            os.path.join(self.model_path, "hmm_spam_model.pkl")
        )
        self.hmm_ham_model = joblib.load(
            os.path.join(self.model_path, "hmm_ham_model.pkl")
        )

    def predict_email(self, email_content, header_spam_score=0.0):
        """
        Predict if an email is spam or ham using all models

        Args:
            email_content (str): The content of the email
            header_spam_score (float): Spam score from email headers

        Returns:
            tuple: (prediction (1 for spam, 0 for ham), probability, detailed results)
        """

        # Preprocess the email content
        processed_content = self.preprocess_text(email_content)

        # Check for suspicious email attributes
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
            "order": 0.2,  # Was likely higher
            "invoice": 0.2,  # Was likely higher
            "payment": 0.2,  # Was likely higher
            "bank": 0.2,  # Was likely higher
            "account": 0.2,  # Was likely higher
            "track": 0.2,
        }

        # Calculate weighted spam score
        content_lower = processed_content.lower()
        spam_score = 0
        total_weight = 0
        for indicator, weight in spam_indicators.items():
            if indicator in content_lower:
                spam_score += weight
                total_weight += weight

        # Normalize indicator score (0 to 1)
        indicator_score = spam_score / (total_weight + 1) if total_weight > 0 else 0

        white_list_terms = {
            "order details": -1.0,
            "invoice": -0.5,
            "tracking": -0.5,
            "amazon.com": -1.0,
            "financial department": -1.0,
            "client": -0.5,
            "shipped": -0.5,  # Add this
            "arrive": -0.5,  # Add this
            "attached": -0.5,  # Add this
            "report": -0.5,  # Add this
            "details": -0.5,  # Add this
            "thank you": -0.5,  # Add this
            "regards": -0.5,  # Add this
            "best regards": -0.5,  # Add this
        }

        # Then apply them to reduce the spam score
        for term, weight in white_list_terms.items():
            if term in content_lower:
                indicator_score += weight

        # Add header score with reduced weight
        indicator_score += header_spam_score / 8.0

        # Vectorize the content for ML models
        content_tfidf = self.vectorizer.transform([processed_content])

        # Get predictions from each model
        nb_pred = self.naive_bayes_model.predict(content_tfidf)[0]
        nb_prob = self.naive_bayes_model.predict_proba(content_tfidf)[0][1]

        svm_pred = self.svm_model.predict(content_tfidf)[0]
        svm_prob = self.svm_model.predict_proba(content_tfidf)[0][1]

        # Get HMM prediction
        try:
            hmm_pred, hmm_prob = self.hmm_predict([processed_content])
            hmm_pred = hmm_pred[0]
            hmm_prob = hmm_prob[0]
        except Exception as e:
            logger.error(f"Error in HMM prediction: {str(e)}")
            hmm_pred = 0
            hmm_prob = 0.0

        # Combine all probabilities (weighted average)
        if nb_prob < 0.5:
            combined_prob = (
                (nb_prob * 0.6)  # Increase NB weight when it says ham
                + (svm_prob * 0.25)  # Reduce SVM weight
                + (hmm_prob * 0.05)  # Almost ignore HMM (it's not useful)
                + (indicator_score * 0.1)
            )
        else:
            # Standard weights when NB thinks it's spam
            combined_prob = (
                (nb_prob * 0.4)
                + (svm_prob * 0.35)
                + (hmm_prob * 0.15)
                + (indicator_score * 0.1)
            )

        # Force ham classification if NB is confident it's ham AND indicator score is negative
        if nb_prob < 0.3 and indicator_score < 0:
            combined_prob = 0.3  # Well below threshold

        # Force ham classification if indicator score is very negative
        if indicator_score < -1.0:
            combined_prob = 0.4  # Below threshold

        # Final prediction based on combined probability
        is_spam = 1 if combined_prob >= 0.5 else 0

        # Detailed results for analysis and visualization
        detailed_results = {
            "is_spam": is_spam,
            "spam_probability": combined_prob,
            "naive_bayes_prediction": nb_pred,
            "naive_bayes_probability": nb_prob,
            "svm_prediction": svm_pred,
            "svm_probability": svm_prob,
            "hmm_prediction": hmm_pred,
            "hmm_probability": hmm_prob,
            "indicator_score": indicator_score,
            "processed_content": processed_content,
        }

        return is_spam, combined_prob, detailed_results

    def visualize_email_prediction(self, email_content, output_path=None):
        """
        Visualize the prediction process for a specific email

        Args:
            email_content: Content of the email to analyze
            output_path: Path to save the visualization (default: visualizations directory)
        """
        if output_path is None:
            output_path = os.path.join(self.visualization_dir, "email_prediction.png")

        # Process the email and get probabilities
        is_spam, combined_prob, details = self.predict_email(email_content)

        # Extract probabilities
        nb_prob = details["naive_bayes_probability"]
        svm_prob = details["svm_probability"]
        hmm_prob = details["hmm_probability"]
        indicator_score = details["indicator_score"]

        # Create a figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

        # Plot 1: Bar chart showing individual model predictions
        models = ["Naive Bayes", "SVM", "HMM", "Combined"]
        probabilities = [nb_prob, svm_prob, hmm_prob, combined_prob]
        colors = ["blue", "red", "purple", "green"]

        ax1.bar(models, probabilities, color=colors, alpha=0.7)
        ax1.set_ylim(0, 1)
        ax1.set_ylabel("Spam Probability")
        ax1.set_title("Model Predictions")
        ax1.axhline(y=0.5, color="black", linestyle="--", label="Decision Threshold")

        # Add probability values on bars
        for i, v in enumerate(probabilities):
            ax1.text(i, v + 0.05, f"{v:.2f}", ha="center")

        # Plot 2: Scatter plot showing relationship between NB and SVM with HMM as color
        ax2.scatter(
            nb_prob,
            svm_prob,
            c=hmm_prob,
            cmap="viridis",
            s=200,
            marker="o",
            edgecolors="black",
        )

        # Add decision quadrants
        ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        ax2.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)

        # Add quadrant labels
        ax2.text(0.25, 0.25, "Both predict HAM", ha="center", va="center", fontsize=10)
        ax2.text(
            0.75, 0.25, "NB: SPAM\nSVM: HAM", ha="center", va="center", fontsize=10
        )
        ax2.text(
            0.25, 0.75, "NB: HAM\nSVM: SPAM", ha="center", va="center", fontsize=10
        )
        ax2.text(0.75, 0.75, "Both predict SPAM", ha="center", va="center", fontsize=10)

        # Add colorbar for HMM probability
        scatter = ax2.scatter(
            [nb_prob],
            [svm_prob],
            c=[hmm_prob],
            cmap="viridis",
            s=200,
            marker="o",
            edgecolors="black",
        )
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label("HMM Probability")

        # Mark the current email's position
        ax2.plot([nb_prob], [svm_prob], "ro", ms=10, mfc="none", mew=2)

        # Add grid and labels
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel("Naive Bayes Probability")
        ax2.set_ylabel("SVM Probability")
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.set_title("Relationship Between Model Probabilities")

        # Add email details as text
        plt.figtext(
            0.5,
            0.01,
            f"Email Classification: {'SPAM' if is_spam else 'HAM'} (Combined Probability: {combined_prob:.2f})",
            ha="center",
            fontsize=12,
            bbox={"facecolor": "lightgray", "alpha": 0.5},
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(output_path, dpi=300)
        plt.close()

        logger.info(f"Email prediction visualization saved to {output_path}")

        return output_path

    def process_email_file(self, file_path):
        # Extract email content and header score
        content, header_spam_score = self.spam_detector.extract_email_content(file_path)

        if not content:
            logger.warning(f"Could not extract content from {file_path}")
            return None

        # Process with spam detector
        result = process_email(content, self.spam_detector)

        # Add debugging logging
        details = result.get("details", {})
        logger.info(f"\nFile: {os.path.basename(file_path)}")
        logger.info(f"NB: {details.get('naive_bayes_probability', 0):.4f}")
        logger.info(f"SVM: {details.get('svm_probability', 0):.4f}")
        logger.info(f"HMM: {details.get('hmm_probability', 0):.4f}")
        logger.info(f"Indicator: {details.get('indicator_score', 0):.4f}")
        logger.info(f"Combined: {result.get('spam_probability', 0):.4f}")
        logger.info(f"Is Spam: {result.get('is_spam', False)}")

        # Generate visualization for this email
        filename = os.path.basename(file_path).replace(".eml", "")
        visualization_path = os.path.join(
            "visualizations", f"{filename}_prediction.png"
        )
        os.makedirs("visualizations", exist_ok=True)

        # Create visualization
        self.spam_detector.visualize_email_prediction(content, visualization_path)
        logger.info(f"Created visualization: {visualization_path}")

        # Add additional info to result
        result["file_path"] = file_path
        result["file_name"] = os.path.basename(file_path)
        result["processed_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result["visualization_path"] = visualization_path

        return result


def process_email(email_content, spam_detector):
    """
    Process an email and detect if it's spam

    Args:
        email_content (str): The content of the email
        spam_detector: Initialized SpamDetector instance

    Returns:
        dict: Detection result with spam status and probability
    """
    # Extract content and header score if email_content is a tuple
    if isinstance(email_content, tuple):
        content, header_spam_score = email_content
    else:
        content = email_content
        header_spam_score = 0.0

    # Get prediction
    is_spam, spam_probability, detailed_results = spam_detector.predict_email(
        content, header_spam_score
    )

    # Prepare result
    result = {
        "is_spam": bool(is_spam),
        "spam_probability": (
            float(spam_probability) if spam_probability is not None else 0.0
        ),
        "details": detailed_results,
    }

    return result


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Spam Detector")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./dataset",
        help="Directory containing spam and ham folders",
    )
    parser.add_argument(
        "--train", action="store_true", help="Force retraining of models"
    )
    args = parser.parse_args()

    # Initialize spam detector
    spam_detector = SpamDetector()

    # Train if models don't exist or if --train flag is set
    if (
        not os.path.exists(
            os.path.join(spam_detector.model_path, "naive_bayes_model.pkl")
        )
        or args.train
    ):
        print("Training models...")
        if not spam_detector.train(args.data_dir):
            print(
                "Training failed. Please check the data directory path and try again."
            )
            exit(1)
    else:
        print("Loading pre-trained models...")
        try:
            spam_detector.load_models()
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            print("Training new models...")
            if not spam_detector.train(args.data_dir):
                print(
                    "Training failed. Please check the data directory path and try again."
                )
                exit(1)

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
    print("\nDetailed Results:")
    for key, value in result["details"].items():
        if key != "processed_content":  # Skip the full processed content
            print(f"  {key}: {value}")
