# ============================================================
#  agent/preprocess.py  —  Text Preprocessing for Fake Job Detection
#  Every line is explained so beginners can follow along easily.
# ============================================================

# ── Imports ──────────────────────────────────────────────────
import re                          # 're' = regular expressions, used to find/remove patterns in text
import nltk                        # Natural Language Toolkit — our main NLP library
from nltk.corpus import stopwords  # Pre-built list of common words like "is", "the", "and"
from nltk.tokenize import word_tokenize  # Splits a sentence into individual words
from nltk.stem import PorterStemmer     # Reduces words to their root form (e.g. "running" → "run")

# ── Download required NLTK data (only needed once) ───────────
# These are small data files NLTK needs locally to work correctly.
# 'quiet=True' means it won't print messages if already downloaded.
nltk.download("punkt",         quiet=True)   # tokenizer rules
nltk.download("punkt_tab",    quiet=True)   # updated tokenizer tables (newer NLTK)
nltk.download("stopwords",    quiet=True)   # the stopword list

# ── Setup ─────────────────────────────────────────────────────
# Load the English stopword list once here so it's not reloaded on every function call.
# STOP_WORDS is a Python set — sets are faster to search than lists.
STOP_WORDS = set(stopwords.words("english"))

# Create a PorterStemmer object.
# Stemming = chopping off word endings to get the base/root word.
# Examples: "working" → "work" | "jobs" → "job" | "easily" → "easili"
stemmer = PorterStemmer()


# ── Main Function ─────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    """
    Clean and normalize a job posting message for ML processing.

    Pipeline:
        raw text
          → lowercase
          → remove URLs
          → remove punctuation & special chars
          → tokenize
          → remove stopwords
          → stem each token
          → rejoin into a clean string

    Args:
        text (str): Raw job message, e.g. "Earn ₹5000/day, No Experience Needed!!"

    Returns:
        str: Cleaned text ready for TF-IDF vectorization, e.g. "earn 5000 dai experi need"
    """

    # ── STEP 1: Lowercase ─────────────────────────────────────
    # Convert everything to lowercase so "Job", "JOB", "job" are treated as the same word.
    # Without this, the model would see them as 3 different words.
    text = text.lower()
    # Example:  "Earn ₹5000/Day NOW!!"  →  "earn ₹5000/day now!!"

    # ── STEP 2: Remove URLs ───────────────────────────────────
    # Scam messages often contain links like "bit.ly/xyz" or "http://fraudsite.com"
    # re.sub(pattern, replacement, string) finds all matches of 'pattern' and replaces with replacement.
    # r"http\S+" matches "http" followed by any non-space characters (\S+)
    text = re.sub(r"http\S+|www\S+", "", text)
    # Example:  "visit http://scam.com now"  →  "visit  now"

    # ── STEP 3: Remove Indian Rupee symbol and currency clutter ──
    # The ₹ symbol and slash combos (₹5000/day) don't help the model — remove them.
    # This regex removes ₹ and / characters.
    text = re.sub(r"[₹/]", " ", text)
    # Example:  "earn ₹5000/day"  →  "earn 5000 day"

    # ── STEP 4: Remove punctuation and special characters ─────
    # Keep only letters (a-z), digits (0-9), and spaces.
    # Everything else — !, @, #, *, (, ), etc. — is replaced with a space.
    # [^a-z0-9\s] means "anything that is NOT a letter, digit, or whitespace"
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Example:  "earn 5000 day, no experience!! call@9876"  →  "earn 5000 day  no experience  call 9876"

    # ── STEP 5: Collapse multiple spaces into one ─────────────
    # After removals, we often get multiple spaces in a row.
    # \s+ matches one or more whitespace characters and replaces with a single space.
    # .strip() removes any leading or trailing spaces from both ends.
    text = re.sub(r"\s+", " ", text).strip()
    # Example:  "earn  5000  day  no"  →  "earn 5000 day no"

    # ── STEP 6: Tokenize — split sentence into list of words ──
    # word_tokenize("earn 5000 day no") → ["earn", "5000", "day", "no"]
    # This gives us individual tokens (words) we can process one by one.
    tokens = word_tokenize(text)

    # ── STEP 7: Remove stopwords + STEP 8: Stem each word ────
    # We do both in a single loop for efficiency (one pass instead of two).
    #
    # Stopwords like "is", "the", "and", "for", "in", "no" carry no useful signal
    # for detecting scams, so we skip them.
    #
    # Stemming reduces words to their root:
    #   "registration" → "registr"
    #   "guaranteed"   → "guarante"
    #   "earning"      → "earn"
    # This helps the model match "earn", "earned", "earnings" as the same concept.
    cleaned_tokens = []
    for token in tokens:
        # Skip the token if it is a stopword
        if token in STOP_WORDS:
            continue
        # Apply stemming and add to our result list
        stemmed = stemmer.stem(token)
        cleaned_tokens.append(stemmed)

    # ── STEP 9: Rejoin tokens into a single string ────────────
    # TF-IDF vectorizer in scikit-learn expects a string, not a list.
    # " ".join(list) puts spaces between each word.
    # Example: ["earn", "5000", "dai", "experi"] → "earn 5000 dai experi"
    cleaned_text = " ".join(cleaned_tokens)

    return cleaned_text


# ── Demo / Test ───────────────────────────────────────────────
if __name__ == "__main__":
    # This block only runs when you execute this file directly:
    #   python preprocess.py
    # It does NOT run when you import this file from another module.

    # Color codes for terminal output (no extra library needed)
    CYAN  = "\033[96m"
    GREEN = "\033[92m"
    RED   = "\033[91m"
    BOLD  = "\033[1m"
    RESET = "\033[0m"

    print(f"\n{BOLD}{CYAN}{'='*60}")
    print("  🧹 Text Preprocessor — Fake Job Detection Agent")
    print(f"{'='*60}{RESET}\n")

    # Sample messages to test — a mix of scam, safe, and Hinglish
    test_messages = [
        ("SCAM",  "Earn ₹5000/Day from Home!! No Experience Needed, WhatsApp NOW: 9876543210"),
        ("SAFE",  "TCS is hiring freshers for Software Engineer role, apply at tcs.com/careers"),
        ("SCAM",  "Ghar baithe ₹50000 kamao!! Registration fee sirf ₹500, abhi WhatsApp karo"),
        ("SAFE",  "Infosys walk-in drive on 15th June in Pune for Java Developer, 2-5 years exp"),
        ("SCAM",  "URGENT!! Pay ₹999 security deposit & start earning ₹3000/day GUARANTEED!!!"),
        ("SAFE",  "Amazon India warehouse associate job in Bhiwandi, ₹18000/month, apply at amazon.jobs"),
    ]

    for label, msg in test_messages:
        result = preprocess_text(msg)
        color = RED if label == "SCAM" else GREEN
        tag   = "🚨 SCAM" if label == "SCAM" else "✅ SAFE"

        print(f"{color}{BOLD}[{tag}]{RESET}")
        print(f"  {BOLD}Raw:    {RESET}{msg}")
        print(f"  {BOLD}Clean:  {RESET}{CYAN}{result}{RESET}")
        print()

    # Show step-by-step breakdown for one message
    print(f"{BOLD}{'─'*60}")
    print("  🔬 Step-by-Step Breakdown")
    print(f"{'─'*60}{RESET}")
    demo = "Earn ₹5000/Day from Home!! No Experience Needed!!"
    print(f"  Original   : {demo}")
    print(f"  Lowercase  : {demo.lower()}")
    s2 = re.sub(r"http\S+|www\S+", "", demo.lower())
    print(f"  No URLs    : {s2}")
    s3 = re.sub(r"[₹/]", " ", s2)
    print(f"  No ₹/slash : {s3}")
    s4 = re.sub(r"[^a-z0-9\s]", " ", s3)
    print(f"  No punct   : {s4}")
    s5 = re.sub(r"\s+", " ", s4).strip()
    print(f"  Clean WS   : {s5}")
    toks = word_tokenize(s5)
    print(f"  Tokens     : {toks}")
    filtered = [stemmer.stem(t) for t in toks if t not in STOP_WORDS]
    print(f"  Stemmed    : {filtered}")
    print(f"  Final      : {' '.join(filtered)}")
    print()