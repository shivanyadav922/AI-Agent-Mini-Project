# 🛡️ AI Agent — Fake Job Detection System

> **College Mini Project Submission**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![ML](https://img.shields.io/badge/ML-TF--IDF%20%2B%20Logistic%20Regression-orange)](https://scikit-learn.org)
[![Domain](https://img.shields.io/badge/Domain-Cybersecurity%20%7C%20AI-red)](https://github.com/shivanyadav922/AI-Agent-Mini-Project)
[![Type](https://img.shields.io/badge/Type-CLI%20AI%20Agent-green)](https://github.com/shivanyadav922/AI-Agent-Mini-Project)

---

## 📌 Project Overview

**AI Agent** is an intelligent CLI-based cybersecurity tool that detects **fake and scam job postings** in the Indian job market. It uses Machine Learning (TF-IDF + Logistic Regression) to analyze job messages in real-time and classify them as **SCAM** or **SAFE** — with confidence scores, keyword threat analysis, and actionable alerts.

This project addresses a critical real-world problem: millions of job seekers in India fall victim to fraudulent job offers every year, especially those circulated via WhatsApp and Telegram.

---

## ✨ Features

- 🤖 **AI-Powered Detection** — TF-IDF vectorization + Logistic Regression classifier
- 🔍 **Keyword Threat Intelligence** — Detects red-flag phrases across 7 threat categories
- 📊 **Confidence Scoring** — Visual confidence bars for SCAM / SAFE probability
- 🚨 **Threat Level System** — LOW / MEDIUM / HIGH / CRITICAL alerts
- 🌐 **Hinglish Support** — Handles Hindi-English mixed language commonly used in scam messages
- 💻 **Dramatic CLI Interface** — Animated scanning, color-coded output, session statistics
- 📈 **Session Tracking** — Live scan counter, scam rate, scan history

---

## 👥 Team & Contributors

This is a **collaborative team project**. Both team members contributed to the design, development, and testing of the system.

| Contributor | GitHub Profile | Role |
|---|---|---|
| Yuvraj Singh Chouhan | [@Yuvraj0772](https://github.com/Yuvraj0772) | ML Model, Data Pipeline, Training Script |
| Shivan Singh Yadav | [@shivanyadav922](https://github.com/shivanyadav922) | Agent CLI, Keyword Engine, Integration & Testing |

> 🔗 **Original Repository:** [https://github.com/Yuvraj0772/AI-Agent](https://github.com/Yuvraj0772/AI-Agent)

---

## 🗂️ Project Structure

```
AI-Agent-Mini-Project/
│
├── agent.py            # Main CLI agent — entry point, detection loop, UI
├── model.py            # FakeJobModel class — TF-IDF + Logistic Regression pipeline
├── preprocess.py       # Text preprocessing — clean, stem, remove stopwords
├── train_model.py      # Model training script — load data, train, evaluate, save
├── setup.py            # NLTK dependency setup
│
├── data/
│   └── jobs.csv        # Labeled training dataset (SCAM=1, SAFE=0)
│
├── model/
│   └── fake_job_model.pkl   # Saved trained model (binary)
│
├── dataset.csv         # Additional dataset
└── scam_logs.txt       # Sample scam message logs
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.8+ | Core language |
| scikit-learn | TF-IDF Vectorizer + Logistic Regression |
| NLTK | Tokenization, stemming, stopword removal |
| pandas | Dataset loading and manipulation |
| numpy | Numerical operations |
| pickle | Model serialization |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install scikit-learn nltk pandas numpy
```

### Installation & Setup

```bash
# Clone the repository
git clone https://github.com/shivanyadav922/AI-Agent-Mini-Project.git
cd AI-Agent-Mini-Project

# Install NLTK data
python setup.py

# Train the model (only needed once)
python train_model.py

# Run the AI Agent
python agent.py
```

---

## 🎮 How to Use

Once the agent is running, paste any job message at the prompt:

```
[#0001] AGENT> Earn ₹5000/day from home!! No experience needed, WhatsApp NOW
```

The agent will respond with:
- ✅ SAFE or 🚨 SCAM verdict
- Confidence percentage
- List of detected risky keywords by category
- Threat level (LOW / MEDIUM / HIGH / CRITICAL)
- Actionable safety advice

### Special Commands

| Command | Action |
|---|---|
| `help` | Show command reference |
| `stats` | Session scan statistics |
| `history` | Last 5 scanned messages |
| `tip` | Random scam awareness tip |
| `clear` | Clear the terminal |
| `exit` | Shut down the agent |

---

## 🧠 How It Works

```
Job Message (raw text)
        │
        ▼
  Preprocessing
  (lowercase → remove URLs → strip punctuation → tokenize → remove stopwords → stem)
        │
        ▼
  TF-IDF Vectorization
  (converts text to numerical feature matrix)
        │
        ▼
  Logistic Regression Classifier
  (outputs probability: SCAM or SAFE)
        │
        ▼
  Keyword Threat Scanner
  (7 threat categories: money traps, fee scams, urgency, etc.)
        │
        ▼
  Combined Result
  (verdict + confidence + threat level + safety advice)
```

---

## 📊 Model Performance

The model is trained on a labeled dataset of Indian job postings:

- **Algorithm:** Logistic Regression with TF-IDF features
- **Vocabulary Size:** ~8,000 unique terms (including bigrams)
- **Language Support:** English + Hinglish
- **Dataset:** Labeled CSV with `message` and `label` columns (0=SAFE, 1=SCAM)

---

## 🔐 Threat Categories Detected

| Category | Examples |
|---|---|
| 💰 Money Trap | "guaranteed income", "earn daily", "₹50000/month" |
| 🔐 Fee / Deposit Scam | "registration fee", "security deposit", "pay first" |
| 🚨 No-Interview Red Flag | "no interview", "direct joining", "no experience needed" |
| 📱 Suspicious Contact | "WhatsApp karo", "send Aadhaar", "Telegram" |
| ⏰ Urgency Pressure | "urgent", "limited seats", "offer expires today" |
| 🏠 Work From Home Lure | "ghar se karo", "WFH", "2 ghante kaam" |
| 🎰 Too-Good-To-Be-True | "100% guaranteed", "risk-free", "lottery", "lucky draw" |

---

## 🤝 Acknowledgements

- Developed as a college mini-project by **Yuvraj Singh Chouhan** and **Shivan Singh Yadav**
- Inspired by real-world job fraud cases targeting Indian youth
- Dataset curated from publicly available scam job message samples

---

## 📄 Academic Submission

> **Project:** AI Agent — Fake Job Detection System  
> **Type:** Mini Project  
> **Domain:** Artificial Intelligence / Cybersecurity  
> **Team:** Yuvraj Singh Chouhan & Shivan Singh Yadav  
> **Repository:** [https://github.com/shivanyadav922/AI-Agent-Mini-Project](https://github.com/shivanyadav922/AI-Agent-Mini-Project)
