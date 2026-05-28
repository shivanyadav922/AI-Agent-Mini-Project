# =============================================================
#  train_model.py  —  The Training Script
#
#  This is the main script you run to train your model.
#  It does the following in order:
#
#    1. Load the CSV dataset
#    2. Preprocess all text (clean, stem, remove stopwords)
#    3. Split data into train and test sets
#    4. Train the FakeJobModel
#    5. Evaluate: Accuracy, Precision, Recall, Confusion Matrix
#    6. Save the trained model to disk
#
#  Run this file with:
#      python train_model.py
# =============================================================

# ── Standard Library ─────────────────────────────────────────
import os
import sys
import time
import pickle

# ── Data Handling ─────────────────────────────────────────────
import pandas as pd           # For loading and working with CSV data (like Excel in Python)
import numpy as np            # Numerical operations (used by sklearn internally)

# ── Machine Learning ─────────────────────────────────────────
from sklearn.model_selection import train_test_split   # Splits data into train & test sets
from sklearn.metrics import (
    accuracy_score,            # % of correct predictions overall
    precision_score,           # Of all predicted SCAMs, how many were actually SCAM?
    recall_score,              # Of all actual SCAMs, how many did we catch?
    f1_score,                  # Harmonic mean of precision & recall (balanced measure)
    confusion_matrix,          # Table showing correct vs incorrect predictions
    classification_report,     # Full text summary of all metrics per class
)

# ── Our Custom Modules ────────────────────────────────────────
# Make sure preprocess.py and model.py are in the same folder
# OR in agent/ subfolder — adjust the import path accordingly.
try:
    from preprocess import preprocess_text   # Our cleaning function
    from model import FakeJobModel           # Our ML model class
except ImportError:
    # If running from project root with agent/ subfolder:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "agent"))
    from preprocess import preprocess_text
    from model import FakeJobModel


# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION  —  Change these paths if needed
# ═══════════════════════════════════════════════════════════════
DATASET_PATH = "data/jobs.csv"          # Path to your CSV dataset
MODEL_SAVE_PATH = "model/fake_job_model.pkl"   # Where to save the trained model
TEST_SIZE = 0.20                         # 20% of data used for testing, 80% for training
RANDOM_STATE = 42                        # Seed for reproducibility (same split every run)

# Terminal color codes — makes output easier to read
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"


# ═══════════════════════════════════════════════════════════════
#  HELPER: Pretty Print Functions
# ═══════════════════════════════════════════════════════════════

def section(title):
    """Prints a formatted section header to the terminal."""
    print(f"\n{BOLD}{CYAN}{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}{RESET}")


def print_confusion_matrix(cm, labels=["SAFE (0)", "SCAM (1)"]):
    """
    Prints a human-readable confusion matrix.

    Confusion Matrix structure:
    ┌──────────────────┬──────────────┬──────────────┐
    │                  │ Predicted: 0 │ Predicted: 1 │
    ├──────────────────┼──────────────┼──────────────┤
    │ Actual: 0 (SAFE) │ TN ✅        │ FP ❌         │
    │ Actual: 1 (SCAM) │ FN ❌        │ TP ✅         │
    └──────────────────┴──────────────┴──────────────┘

    TN = True Negative  : Correctly predicted SAFE
    FP = False Positive : Wrongly flagged SAFE as SCAM
    FN = False Negative : Missed a SCAM (most dangerous!)
    TP = True Positive  : Correctly caught SCAM
    """
    tn, fp, fn, tp = cm.ravel()   # .ravel() flattens 2D array into 4 values

    print(f"\n  {BOLD}Confusion Matrix:{RESET}")
    print(f"  {'':20s} {'Pred: SAFE':>12} {'Pred: SCAM':>12}")
    print(f"  {'─'*46}")
    print(f"  {'Actual: SAFE (0)':20s} {GREEN}{tn:>12}{RESET} {RED}{fp:>12}{RESET}")
    print(f"  {'Actual: SCAM (1)':20s} {RED}{fn:>12}{RESET} {GREEN}{tp:>12}{RESET}")
    print(f"  {'─'*46}")
    print(f"\n  {GREEN}✅ TN (Correctly SAFE) : {tn}{RESET}  — Model said SAFE, it was SAFE")
    print(f"  {RED}❌ FP (False Alarm)    : {fp}{RESET}  — Model said SCAM, it was SAFE")
    print(f"  {RED}🚨 FN (Missed SCAM)    : {fn}{RESET}  — Model said SAFE, it was SCAM ← DANGEROUS!")
    print(f"  {GREEN}✅ TP (Caught SCAM)    : {tp}{RESET}  — Model said SCAM, it was SCAM")


# ═══════════════════════════════════════════════════════════════
#  STEP 1: Load Dataset
# ═══════════════════════════════════════════════════════════════

def load_dataset(path):
    """
    Load the CSV file into a pandas DataFrame and validate it.

    A DataFrame is like a table in Python:
        message                        label
        "Earn ₹5000/day..."            1
        "TCS walk-in drive..."         0
        ...

    Args:
        path (str): Path to the CSV file

    Returns:
        pd.DataFrame: Cleaned DataFrame with 'message' and 'label' columns
    """
    section("STEP 1: Loading Dataset")

    # Check the file actually exists before trying to open it
    if not os.path.exists(path):
        print(f"{RED}[ERROR] Dataset not found at: {path}{RESET}")
        print(f"{YELLOW}  → Create data/jobs.csv with columns: message, label{RESET}")
        sys.exit(1)   # Exit the program with error code 1

    # pd.read_csv() reads the CSV into a DataFrame
    df = pd.read_csv(path)
    print(f"  [+] Loaded file: {path}")
    print(f"  [+] Raw rows found: {len(df)}")

    # ── Validate columns exist ────────────────────────────────
    required_cols = {"message", "label"}
    if not required_cols.issubset(df.columns):
        print(f"{RED}[ERROR] CSV must have columns: 'message' and 'label'{RESET}")
        print(f"  Found: {list(df.columns)}")
        sys.exit(1)

    # ── Drop rows with missing values ─────────────────────────
    # dropna() removes any row where message OR label is empty/NaN
    before = len(df)
    df = df.dropna(subset=["message", "label"])
    dropped = before - len(df)
    if dropped:
        print(f"  {YELLOW}[!] Dropped {dropped} rows with missing values{RESET}")

    # ── Ensure label column contains only 0 and 1 ────────────
    df["label"] = df["label"].astype(int)    # Convert to integer (in case they're strings)
    invalid = df[~df["label"].isin([0, 1])]
    if len(invalid):
        print(f"  {YELLOW}[!] Removed {len(invalid)} rows with invalid labels (not 0 or 1){RESET}")
        df = df[df["label"].isin([0, 1])]

    # ── Print class distribution ──────────────────────────────
    scam_count = (df["label"] == 1).sum()
    safe_count = (df["label"] == 0).sum()
    print(f"  [+] SAFE messages (label=0): {GREEN}{safe_count}{RESET}")
    print(f"  [+] SCAM messages (label=1): {RED}{scam_count}{RESET}")
    print(f"  [+] Total usable rows      : {BOLD}{len(df)}{RESET}")

    return df


# ═══════════════════════════════════════════════════════════════
#  STEP 2: Preprocess All Text
# ═══════════════════════════════════════════════════════════════

def preprocess_dataset(df):
    """
    Apply preprocess_text() to every message in the dataset.

    pandas .apply() runs a function on every row of a column.
    It's like doing a for loop but faster:
        df["clean"] = [preprocess_text(msg) for msg in df["message"]]

    Args:
        df (pd.DataFrame): DataFrame with 'message' column

    Returns:
        pd.DataFrame: Same DataFrame with new 'clean_message' column added
    """
    section("STEP 2: Preprocessing Text")

    print("  [*] Applying text cleaning to all messages...")
    print(f"  {DIM}(lowercasing → remove URLs → remove punctuation → "
          f"tokenize → remove stopwords → stem){RESET}")

    start = time.time()

    # Apply our preprocess_text function to every row in the 'message' column
    # The result is stored in a new column 'clean_message'
    df["clean_message"] = df["message"].apply(preprocess_text)

    elapsed = time.time() - start
    print(f"  [+] Done! Processed {len(df)} messages in {elapsed:.2f}s")

    # Show before/after examples to verify it's working
    print(f"\n  {BOLD}Sample transformations:{RESET}")
    samples = df.sample(min(3, len(df)), random_state=42)
    for _, row in samples.iterrows():
        tag = f"{RED}SCAM{RESET}" if row["label"] == 1 else f"{GREEN}SAFE{RESET}"
        print(f"\n  [{tag}]")
        print(f"    Before: {row['message'][:80]}...")
        print(f"    After : {CYAN}{row['clean_message'][:80]}{RESET}")

    return df


# ═══════════════════════════════════════════════════════════════
#  STEP 3: Split into Train and Test Sets
# ═══════════════════════════════════════════════════════════════

def split_data(df):
    """
    Split data into training set and test set.

    Why split?
        - We train on 80% of data so the model LEARNS patterns
        - We test on the other 20% — data the model has NEVER seen
        - This tells us how well the model generalizes to new job messages
        - If we tested on training data, the model could just memorize answers!

    stratify=y ensures both train and test sets have the same
    ratio of SCAM/SAFE messages. Without it, all SCAMs could end
    up in one set by chance.

    Args:
        df (pd.DataFrame): DataFrame with 'clean_message' and 'label' columns

    Returns:
        X_train, X_test, y_train, y_test (4 separate sets)
    """
    section("STEP 3: Splitting Train / Test Data")

    # X = features (the text we use to make predictions)
    # y = labels (the correct answers: 0 or 1)
    X = df["clean_message"]
    y = df["label"]

    # train_test_split() randomly shuffles and splits the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,       # 0.20 = 20% goes to test set
        random_state=RANDOM_STATE, # Ensures same split every time
        stratify=y,                # Maintain SCAM/SAFE ratio in both sets
    )

    print(f"  [+] Training set  : {BOLD}{len(X_train)}{RESET} messages  "
          f"(SCAM: {(y_train==1).sum()}, SAFE: {(y_train==0).sum()})")
    print(f"  [+] Test set      : {BOLD}{len(X_test)}{RESET} messages   "
          f"(SCAM: {(y_test==1).sum()}, SAFE: {(y_test==0).sum()})")
    print(f"  [+] Split ratio   : {int((1-TEST_SIZE)*100)}% train / {int(TEST_SIZE*100)}% test")

    return X_train, X_test, y_train, y_test


# ═══════════════════════════════════════════════════════════════
#  STEP 4: Train the Model
# ═══════════════════════════════════════════════════════════════

def train(X_train, y_train):
    """
    Create a FakeJobModel and train it on the training data.

    During training the model:
        1. TF-IDF learns the vocabulary from all training messages
        2. TF-IDF converts all messages to number arrays
        3. Logistic Regression finds the best weights for each word

    After training, the model "knows" which words/phrases
    are most associated with scam vs. safe jobs.

    Args:
        X_train: Training messages (strings)
        y_train: Training labels (0 or 1)

    Returns:
        FakeJobModel: Trained model ready to make predictions
    """
    section("STEP 4: Training the Model")

    model = FakeJobModel()   # Create a fresh untrained model

    print("  [*] Training TF-IDF + Logistic Regression pipeline...")
    print(f"  {DIM}(This builds the vocabulary and finds the best word weights){RESET}")

    start = time.time()
    model.train(X_train, y_train)    # ← The actual training happens here
    elapsed = time.time() - start

    print(f"  [+] Training complete in {elapsed:.2f}s")
    print(f"  [+] Vocabulary size: {len(model.pipeline['tfidf'].vocabulary_)} unique terms")

    return model


# ═══════════════════════════════════════════════════════════════
#  STEP 5: Evaluate the Model
# ═══════════════════════════════════════════════════════════════

def evaluate(model, X_test, y_test):
    """
    Measure how well the model performs on unseen test data.

    Metrics explained:
    ┌────────────────┬────────────────────────────────────────────────┐
    │ Accuracy       │ % of all predictions that were correct         │
    │                │ Limitation: misleading if SCAM/SAFE unbalanced │
    ├────────────────┼────────────────────────────────────────────────┤
    │ Precision      │ Of all predicted SCAMs, how many were real?    │
    │                │ High precision = fewer false alarms             │
    ├────────────────┼────────────────────────────────────────────────┤
    │ Recall         │ Of all actual SCAMs, how many did we catch?    │
    │                │ High recall = fewer missed scams (critical!)    │
    ├────────────────┼────────────────────────────────────────────────┤
    │ F1 Score       │ Balance between precision and recall           │
    │                │ Best single metric for imbalanced data          │
    └────────────────┴────────────────────────────────────────────────┘

    Args:
        model: Trained FakeJobModel
        X_test: Test messages
        y_test: True labels for test set
    """
    section("STEP 5: Evaluating Model Performance")

    # Get predictions for all test messages
    y_pred = model.predict(X_test)

    # ── Core Metrics ──────────────────────────────────────────
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)   # defaults to positive class (SCAM=1)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    cm        = confusion_matrix(y_test, y_pred)

    # ── Print Metrics ─────────────────────────────────────────
    def score_color(val):
        """Color-code a score: green if good, yellow if ok, red if bad."""
        if val >= 0.85: return GREEN
        if val >= 0.70: return YELLOW
        return RED

    print(f"\n  {BOLD}Core Metrics (on {len(X_test)} unseen messages):{RESET}")
    print(f"  {'Accuracy' :<15} {score_color(accuracy )}{accuracy :.4f}  ({accuracy *100:.1f}%){RESET}")
    print(f"  {'Precision':<15} {score_color(precision)}{precision:.4f}  ({precision*100:.1f}%){RESET}")
    print(f"  {'Recall'   :<15} {score_color(recall   )}{recall   :.4f}  ({recall   *100:.1f}%){RESET}")
    print(f"  {'F1 Score' :<15} {score_color(f1       )}{f1       :.4f}  ({f1       *100:.1f}%){RESET}")

    # ── Confusion Matrix ──────────────────────────────────────
    print_confusion_matrix(cm)

    # ── Full sklearn Classification Report ───────────────────
    print(f"\n  {BOLD}Full Classification Report:{RESET}")
    report = classification_report(
        y_test, y_pred,
        target_names=["SAFE (0)", "SCAM (1)"]
    )
    # Indent each line of the report for cleaner output
    for line in report.splitlines():
        print(f"  {line}")

    # ── Show incorrectly predicted examples ──────────────────
    print(f"\n  {BOLD}Misclassified Examples (first 3):{RESET}")
    # Reset index so we can use positional indexing
    X_test_reset = X_test.reset_index(drop=True)
    y_test_reset = y_test.reset_index(drop=True)

    wrong_mask = y_pred != y_test_reset.values
    wrong_indices = [i for i, w in enumerate(wrong_mask) if w]

    if not wrong_indices:
        print(f"  {GREEN}🎉 No misclassifications! Perfect test set accuracy.{RESET}")
    else:
        for idx in wrong_indices[:3]:
            true_label = "SCAM" if y_test_reset.iloc[idx] == 1 else "SAFE"
            pred_label = "SCAM" if y_pred[idx] == 1 else "SAFE"
            print(f"\n  Text   : {X_test_reset.iloc[idx][:70]}...")
            print(f"  {GREEN}True   : {true_label}{RESET}  |  {RED}Predicted: {pred_label}{RESET}")

    return {
        "accuracy" : accuracy,
        "precision": precision,
        "recall"   : recall,
        "f1"       : f1,
        "confusion_matrix": cm,
    }


# ═══════════════════════════════════════════════════════════════
#  STEP 6: Save the Model
# ═══════════════════════════════════════════════════════════════

def save_model(model, path):
    """
    Save the trained model to disk using pickle.

    pickle converts Python objects into bytes and writes them to a file.
    Later, pickle.load() reads the file and reconstructs the Python object.

    This means you only train ONCE — after that, load the saved model
    and use it for predictions without retraining.

    Args:
        model: Trained FakeJobModel
        path: File path to save to (e.g. "model/fake_job_model.pkl")
    """
    section("STEP 6: Saving Model to Disk")
    model.save(path)
    file_size_kb = os.path.getsize(path) / 1024
    print(f"  [+] File size     : {file_size_kb:.1f} KB")
    print(f"  [+] Load it later : {CYAN}FakeJobModel.load('{path}'){RESET}")


# ═══════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def main():
    """
    Runs all 6 training steps in sequence.
    Called automatically when you run: python train_model.py
    """
    # ── Banner ────────────────────────────────────────────────
    print(f"\n{BOLD}{CYAN}{'═'*55}")
    print("  🛡️  FAKE JOB DETECTION AGENT — Model Trainer")
    print(f"{'═'*55}{RESET}")
    print(f"  Dataset  : {DATASET_PATH}")
    print(f"  Save to  : {MODEL_SAVE_PATH}")
    print(f"  Split    : {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} train/test")

    overall_start = time.time()

    # ── Run all steps ─────────────────────────────────────────
    df                              = load_dataset(DATASET_PATH)
    df                              = preprocess_dataset(df)
    X_train, X_test, y_train, y_test = split_data(df)
    model                           = train(X_train, y_train)
    metrics                         = evaluate(model, X_test, y_test)
    save_model(model, MODEL_SAVE_PATH)

    # ── Final Summary ─────────────────────────────────────────
    total_time = time.time() - overall_start
    section("✅ TRAINING COMPLETE")
    print(f"  Total time  : {total_time:.2f}s")
    print(f"  Accuracy    : {GREEN}{metrics['accuracy']*100:.1f}%{RESET}")
    print(f"  F1 Score    : {GREEN}{metrics['f1']*100:.1f}%{RESET}")
    print(f"  Model saved : {MODEL_SAVE_PATH}")
    print(f"\n  {BOLD}Next step:{RESET}")
    print(f"  Run your CLI agent → {CYAN}python cli.py{RESET}\n")


# This ensures main() only runs when you execute this file directly.
# It does NOT run when another file imports from this module.
if __name__ == "__main__":
    main()