import os
import shutil
import email
from datetime import datetime
from run_spam_detector import SpamDetector, process_email
import logging
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LocalEmailProcessor:
    def __init__(
        self,
        input_dir="input_emails",
        spam_dir="processed_emails/spam",
        ham_dir="processed_emails/ham",
        threshold=0.7,
    ):
        """
        Initialize the local email processor.

        Args:
            input_dir: Directory containing input email files
            spam_dir: Directory to move spam emails to
            ham_dir: Directory to move ham (non-spam) emails to
            threshold: Probability threshold for spam classification
        """
        self.input_dir = input_dir
        self.spam_dir = spam_dir
        self.ham_dir = ham_dir
        self.threshold = threshold
        self.spam_detector = SpamDetector()

        # Create directories if they don't exist
        for directory in [input_dir, spam_dir, ham_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"Created directory: {directory}")

    def load_models(self):
        """
        Load the pre-trained spam detection models.
        """
        try:
            self.spam_detector.load_models()
            logger.info("Pre-trained models loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            logger.info(
                "Please train the models first by running the email_spam_detector.py script."
            )
            return False

    def create_summary_visualization(self, results):
        """Create a summary visualization showing all emails classified"""
        if not results:
            logger.warning("No results to visualize")
            return None

        # Extract data
        filenames = []
        nb_probs = []
        svm_probs = []
        hmm_probs = []
        combined_probs = []
        is_spams = []

        for result in results:
            filenames.append(os.path.basename(result["file_path"]))
            details = result.get("details", {})
            nb_probs.append(details.get("naive_bayes_probability", 0))
            svm_probs.append(details.get("svm_probability", 0))
            hmm_probs.append(details.get("hmm_probability", 0))
            combined_probs.append(result.get("spam_probability", 0))
            is_spams.append(result.get("is_spam", False))

        # Create visualization
        plt.figure(figsize=(15, 10))

        # Use a colormap based on spam/ham classification
        colors = ["green" if not is_spam else "red" for is_spam in is_spams]

        # Create scatter plot
        plt.scatter(nb_probs, svm_probs, c=hmm_probs, cmap="viridis", s=200, alpha=0.7)

        # Add labels
        for i, filename in enumerate(filenames):
            plt.annotate(filename, (nb_probs[i], svm_probs[i]), fontsize=8)

        # Add decision lines
        plt.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        plt.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)

        # Add quadrant labels
        plt.text(0.25, 0.25, "Both predict HAM", ha="center", va="center", fontsize=12)
        plt.text(
            0.75, 0.25, "NB: SPAM\nSVM: HAM", ha="center", va="center", fontsize=12
        )
        plt.text(
            0.25, 0.75, "NB: HAM\nSVM: SPAM", ha="center", va="center", fontsize=12
        )
        plt.text(0.75, 0.75, "Both predict SPAM", ha="center", va="center", fontsize=12)

        # Add colorbar for HMM probability
        scatter = plt.scatter(
            nb_probs, svm_probs, c=hmm_probs, cmap="viridis", s=200, alpha=0.7
        )
        cbar = plt.colorbar(scatter)
        cbar.set_label("HMM Probability")

        # Add grid and labels
        plt.grid(True, alpha=0.3)
        plt.xlabel("Naive Bayes Probability")
        plt.ylabel("SVM Probability")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title("Email Classification Overview")

        # Save the visualization
        os.makedirs("visualizations", exist_ok=True)
        visualization_path = os.path.join(
            "visualizations", "classification_summary.png"
        )
        plt.savefig(visualization_path, dpi=300)
        plt.close()

        logger.info(f"Summary visualization saved to {visualization_path}")
        return visualization_path

    def process_email_file(self, file_path):
        # Extract email content and header score
        content, header_spam_score = self.spam_detector.extract_email_content(file_path)

        if not content:
            logger.warning(f"Could not extract content from {file_path}")
            return None

        # Process with spam detector
        result = process_email(content, self.spam_detector)

        # Add this debug logging
        details = result.get("details", {})
        logger.info(f"\nFile: {os.path.basename(file_path)}")
        logger.info(f"NB: {details.get('naive_bayes_probability', 0):.4f}")
        logger.info(f"SVM: {details.get('svm_probability', 0):.4f}")
        logger.info(f"HMM: {details.get('hmm_probability', 0):.4f}")
        logger.info(f"Indicator: {details.get('indicator_score', 0):.4f}")
        logger.info(f"Combined: {result.get('spam_probability', 0):.4f}")
        logger.info(f"Is Spam: {result.get('is_spam', False)}")

        # Add additional info to result
        result["file_path"] = file_path
        result["file_name"] = os.path.basename(file_path)
        result["processed_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return result

    def move_email_file(self, file_path, is_spam):
        """
        Move the email file to the appropriate directory.

        Args:
            file_path: Path to the email file
            is_spam: Boolean indicating if the email is spam

        Returns:
            str: New file path
        """
        file_name = os.path.basename(file_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_file_name = f"{timestamp}_{file_name}"

        if is_spam:
            destination = os.path.join(self.spam_dir, new_file_name)
        else:
            destination = os.path.join(self.ham_dir, new_file_name)

        shutil.copy2(file_path, destination)
        os.remove(file_path)  # Remove from input directory

        return destination

    def process_all_emails(self):
        """
        Process all email files in the input directory.

        Returns:
            tuple: (spam_count, ham_count)
        """
        if not os.path.exists(self.input_dir):
            logger.error(f"Input directory {self.input_dir} does not exist.")
            return 0, 0

        # Get all files in the input directory
        files = [
            f
            for f in os.listdir(self.input_dir)
            if os.path.isfile(os.path.join(self.input_dir, f))
            and (f.endswith(".eml") or f.endswith(".txt") or f.endswith(".msg"))
        ]

        if not files:
            logger.info(f"No email files found in {self.input_dir}")
            return 0, 0

        logger.info(f"Found {len(files)} email files to process.")

        spam_count = 0
        ham_count = 0

        results = []

        # Process each file
        for file_name in files:
            file_path = os.path.join(self.input_dir, file_name)
            logger.info(f"Processing {file_path}")

            result = self.process_email_file(file_path)

            if result:
                results.append(result)  # Store the result
                is_spam = result["is_spam"]
                probability = result["spam_probability"]

                # Move the file to the appropriate directory
                new_path = self.move_email_file(file_path, is_spam)

                if is_spam:
                    spam_count += 1
                    logger.info(f"SPAM ({probability:.4f}): {file_name} -> {new_path}")
                else:
                    ham_count += 1
                    logger.info(f"HAM ({probability:.4f}): {file_name} -> {new_path}")

        if results:
            self.create_summary_visualization(results)

        logger.info(f"Processing complete. Spam: {spam_count}, Ham: {ham_count}")
        return spam_count, ham_count

    def create_test_emails(self, num_samples=10):
        """
        Create test email files for demonstration purposes.

        Args:
            num_samples: Number of test emails to create

        Returns:
            list: Paths to created email files
        """
        if not os.path.exists(self.input_dir):
            os.makedirs(self.input_dir)

        created_files = []

        # Create some spam examples
        spam_examples = [
            {
                "subject": "URGENT: You've WON $5,000,000.00!",
                "body": """
                Dear Lucky Winner,
                
                Congratulations! You have been selected as the winner of our INTERNATIONAL LOTTERY.
                You have won the sum of $5,000,000.00 (Five Million Dollars).
                
                To claim your prize, please send us your:
                1. Full Name
                2. Address
                3. Phone Number
                4. Copy of ID
                5. Bank Account Information
                
                Contact our agent immediately: agent@scam-lottery.com
                
                RESPOND IMMEDIATELY!!!
                """,
            },
            {
                "subject": "Enhance Your Manhood - Special Discount!!",
                "body": """
                SPECIAL OFFER - 90% OFF - TODAY ONLY!!!
                
                Our revolutionary pills will enhance your manhood by 5 inches in just 7 days!
                
                No prescription needed!
                Discreet packaging!
                Money back guarantee!
                
                Click here to order now: www.fake-pills.com
                
                Hurry, offer ends today!
                """,
            },
            {
                "subject": "Re: Your Payment of $499.99",
                "body": """
                Dear Customer,
                
                We have processed your payment of $499.99 for Premium Subscription.
                
                If you did not authorize this transaction, please update your account 
                information immediately by clicking on the link below:
                
                www.secure-account-verify.com/update
                
                Thank you for your attention to this matter.
                
                Customer Service
                """,
            },
            {
                "subject": "ATTENTION: Account Suspension Notice",
                "body": """
                Dear Valued Customer,
                
                Your account has been flagged for suspicious activity and will be suspended 
                within 24 hours.
                
                To prevent suspension, verify your identity by clicking the link below:
                
                http://verify-account-now.com/secure
                
                Failure to verify will result in permanent account closure.
                
                Security Department
                """,
            },
            {
                "subject": "LAST CHANCE - 85% off Luxury Watches",
                "body": """
                *** EXCLUSIVE OFFER ***
                
                Authentic Luxury Watches at 85% OFF RETAIL PRICE!
                
                • Rolex
                • Omega
                • TAG Heuer
                • Cartier
                
                All watches are 100% authentic guaranteed!
                Free shipping worldwide!
                
                CLICK HERE TO SHOP NOW: www.luxury-replica-watches.com
                
                This offer expires in 24 hours!
                """,
            },
        ]

        # Create some legitimate (ham) examples
        ham_examples = [
            {
                "subject": "Team Meeting - Tuesday 10 AM",
                "body": """
                Hi Team,
                
                Let's meet on Tuesday at 10 AM to discuss the quarterly results.
                
                Agenda:
                1. Q1 Performance Review
                2. Q2 Strategy Planning
                3. Budget Updates
                
                Please prepare your department reports.
                
                Regards,
                John
                """,
            },
            {
                "subject": "Your Amazon Order #112-9384756-1294113",
                "body": """
                Hello,
                
                Thank you for your order from Amazon.com.
                
                Your order #112-9384756-1294113 has been shipped and will arrive on Tuesday.
                
                Order Details:
                - JavaScript: The Definitive Guide
                - USB-C Charging Cable (2-pack)
                
                You can track your package here: amazon.com/track
                
                Thank you for shopping with Amazon!
                """,
            },
            {
                "subject": "Dinner this weekend?",
                "body": """
                Hey Sarah,
                
                Are you free for dinner this weekend? We could try that new Italian place
                that opened downtown.
                
                Let me know what day works best for you!
                
                Cheers,
                Michael
                """,
            },
            {
                "subject": "Project Update - Website Redesign",
                "body": """
                Hi everyone,
                
                I wanted to share a quick update on the website redesign project:
                
                - Homepage mockups are complete
                - User testing scheduled for next week
                - Launch date is still on track for June 15
                
                Please review the latest designs in the shared folder and provide feedback
                by Thursday.
                
                Thanks,
                Emma
                Project Manager
                """,
            },
            {
                "subject": "Invoice #2023-042",
                "body": """
                Dear Client,
                
                Please find attached Invoice #2023-042 for services provided in April 2023.
                
                Amount Due: $1,250.00
                Due Date: May 15, 2023
                
                Payment can be made via bank transfer using the account details on the invoice.
                
                If you have any questions, please don't hesitate to contact me.
                
                Best regards,
                Robert Johnson
                Financial Department
                """,
            },
        ]

        # Determine how many of each to create
        spam_count = num_samples // 2
        ham_count = num_samples - spam_count

        # Create spam emails
        for i in range(spam_count):
            example = spam_examples[i % len(spam_examples)]
            file_name = f"spam_sample_{i+1}.eml"
            file_path = os.path.join(self.input_dir, file_name)

            # Create email in RFC 5322 format
            msg = f"""From: spammer@example.com
To: victim@example.com
Subject: {example['subject']}
Date: {datetime.now().strftime('%a, %d %b %Y %H:%M:%S +0000')}
Content-Type: text/plain; charset="UTF-8"

{example['body']}
"""

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(msg)

            created_files.append(file_path)

        # Create ham emails
        for i in range(ham_count):
            example = ham_examples[i % len(ham_examples)]
            file_name = f"ham_sample_{i+1}.eml"
            file_path = os.path.join(self.input_dir, file_name)

            # Create email in RFC 5322 format
            msg = f"""From: colleague@example.com
To: me@example.com
Subject: {example['subject']}
Date: {datetime.now().strftime('%a, %d %b %Y %H:%M:%S +0000')}
Content-Type: text/plain; charset="UTF-8"

{example['body']}
"""

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(msg)

            created_files.append(file_path)

        logger.info(
            f"Created {len(created_files)} test email files in {self.input_dir}"
        )
        return created_files


if __name__ == "__main__":
    processor = LocalEmailProcessor()

    # Check if models are loaded, if not, notify the user
    if not processor.load_models():
        logger.error("Could not load spam detection models. Please train them first.")
        exit(1)

    # Create test emails if no emails exist in the input directory
    input_files = [
        f
        for f in os.listdir(processor.input_dir)
        if os.path.isfile(os.path.join(processor.input_dir, f)) and f.endswith(".eml")
    ]

    if not input_files:
        logger.info("No email files found in input directory. Creating test samples...")
        processor.create_test_emails(10)  # Create 10 test emails

    # Process all emails
    spam_count, ham_count = processor.process_all_emails()

    logger.info(f"Processing complete!")
    logger.info(f"Spam emails: {spam_count}")
    logger.info(f"Ham emails: {ham_count}")
    logger.info(f"Spam emails moved to: {processor.spam_dir}")
    logger.info(f"Ham emails moved to: {processor.ham_dir}")
