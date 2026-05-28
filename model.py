# =============================================================
#  model.py  —  The ML Model Blueprint
#
#  This file defines the FakeJobModel class.
#  Think of it like a blueprint/template that:
#    1. Holds the TF-IDF vectorizer (converts text → numbers)
#    2. Holds the Logistic Regression classifier (makes predictions)
#    3. Has methods to train, predict, and save/load itself
#
#  We keep the model logic here so train_model.py stays clean.
# =============================================================

# ── Imports ──────────────────────────────────────────────────
import pickle                                        # Saves Python objects to disk (like a file)
import os                                            # For file path operations

from sklearn.feature_extraction.text import TfidfVectorizer   # Text → numbers converter
from sklearn.linear_model import LogisticRegression            # Our classifier
from sklearn.pipeline import Pipeline                          # Chains steps into one object


class FakeJobModel:
    """
    A wrapper class that bundles TF-IDF + Logistic Regression into one clean object.

    Why use a class?
        - Keeps vectorizer and classifier always together (no mismatch bugs)
        - Easy to save/load as a single file
        - Clean interface: just call .train(), .predict(), .save()

    Analogy: Think of this as a "black box":
        Input:  raw job text (string)
        Output: "SCAM" or "SAFE" + confidence score
    """

    def __init__(self):
        # ── Build a Pipeline ──────────────────────────────────
        # sklearn's Pipeline chains multiple steps together.
        # When you call pipeline.fit(X, y), it runs all steps in order.
        # When you call pipeline.predict(X), it also runs all steps in order.
        #
        # Our pipeline has 2 steps:
        #   Step 1: "tfidf"   → TfidfVectorizer
        #   Step 2: "clf"     → LogisticRegression
        #
        # Each step is a tuple: ("name", object)
        self.pipeline = Pipeline([

            # ── Step 1: TF-IDF Vectorizer ─────────────────────
            # TF-IDF = Term Frequency × Inverse Document Frequency
            #
            # What it does: converts text strings into number arrays
            # so the ML model can understand them.
            #
            # Example:
            #   "earn money fast" → [0.0, 0.85, 0.0, 0.72, 0.0, 0.61, ...]
            #
            # Parameters explained:
            ("tfidf", TfidfVectorizer(
                max_features=8000,      # Only keep the 8000 most important words
                                        # (ignores very rare words that add noise)

                ngram_range=(1, 2),     # Use single words AND 2-word pairs
                                        # e.g. "work from home" captures:
                                        #   unigrams: "work", "from", "home"
                                        #   bigrams:  "work from", "from home"
                                        # Bigrams are great for catching scam phrases!

                sublinear_tf=True,      # Apply log scaling to term frequency
                                        # Prevents common words from dominating
                                        # Formula: 1 + log(tf) instead of raw tf

                min_df=2,               # Ignore words that appear in fewer than 2 docs
                                        # Removes typos and ultra-rare noise words
            )),

            # ── Step 2: Logistic Regression Classifier ────────
            # Despite the name, Logistic Regression is used for CLASSIFICATION.
            #
            # How it works (simplified):
            #   - It learns a "weight" for every word/phrase
            #   - Scam words get positive weights (e.g. "guaranteed income" = +5)
            #   - Normal words get negative/neutral weights (e.g. "experience required" = -3)
            #   - It sums up all weights for a message and outputs a probability 0–1
            #
            # Parameters explained:
            ("clf", LogisticRegression(
                max_iter=1000,          # Maximum training iterations
                                        # 1000 ensures it fully converges (stops changing)

                C=1.0,                  # Regularization strength (controls overfitting)
                                        # Lower C = simpler model (less overfit)
                                        # Higher C = complex model (fits training data more)
                                        # 1.0 is a well-balanced default

                solver="lbfgs",         # The math algorithm used to find optimal weights
                                        # "lbfgs" is fast, memory-efficient, works well for text

                random_state=42,        # Makes results reproducible (same output every run)
                                        # 42 is just a convention (any number works)
            )),
        ])

        # Flag to track whether the model has been trained yet
        # We check this before allowing predictions
        self.is_trained = False

    def train(self, X_train, y_train):
        """
        Train the model on preprocessed text data.

        Args:
            X_train: List/Series of preprocessed job message strings
            y_train: List/Series of labels (0 = SAFE, 1 = SCAM)

        The pipeline automatically:
            1. Fits the TF-IDF vectorizer on X_train (learns vocabulary)
            2. Transforms X_train into TF-IDF matrix
            3. Trains Logistic Regression on that matrix
        """
        self.pipeline.fit(X_train, y_train)
        self.is_trained = True   # Mark as trained so predict() knows it's ready

    def predict(self, texts):
        """
        Predict SCAM (1) or SAFE (0) for a list of preprocessed messages.

        Args:
            texts: List of preprocessed text strings

        Returns:
            numpy array of predictions: 0 (SAFE) or 1 (SCAM)
        """
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet. Call train() first.")
        return self.pipeline.predict(texts)

    def predict_proba(self, texts):
        """
        Return probability scores for each class.

        Args:
            texts: List of preprocessed text strings

        Returns:
            numpy array of shape (n_samples, 2)
            Column 0 = probability of SAFE
            Column 1 = probability of SCAM

        Example:
            [[0.12, 0.88]] means 12% SAFE, 88% SCAM → likely SCAM
        """
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet. Call train() first.")
        return self.pipeline.predict_proba(texts)

    def save(self, path="model/fake_job_model.pkl"):
        """
        Save the entire trained model (pipeline) to disk using pickle.

        pickle.dump() serializes (converts) a Python object into bytes
        and writes it to a file. Later, pickle.load() reverses this.

        'wb' mode = write binary (required for pickle)

        Args:
            path: File path where the model will be saved
        """
        # Create the directory if it doesn't exist yet
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "wb") as f:         # open file in write-binary mode
            pickle.dump(self.pipeline, f)   # serialize and save the pipeline

        print(f"[✓] Model saved → {path}")

    @classmethod
    def load(cls, path="model/fake_job_model.pkl"):
        """
        Load a previously saved model from disk.

        @classmethod means you call this on the class, not an instance:
            model = FakeJobModel.load("model/fake_job_model.pkl")

        'rb' mode = read binary (required for pickle)

        Args:
            path: File path to the saved .pkl model file

        Returns:
            FakeJobModel instance with the loaded pipeline
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at '{path}'. Train the model first.")

        # Create a blank FakeJobModel instance
        instance = cls()

        with open(path, "rb") as f:              # open file in read-binary mode
            instance.pipeline = pickle.load(f)   # deserialize and restore the pipeline

        instance.is_trained = True
        print(f"[✓] Model loaded ← {path}")
        return instance