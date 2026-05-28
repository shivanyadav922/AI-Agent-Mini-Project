# =============================================================
#  cli.py  —  Fake Job Detection AI Agent (CLI)
#
#  This is the main entry point. Run it with:
#      python cli.py
#
#  It behaves like a cybersecurity terminal agent:
#    - Boots up with a dramatic intro
#    - Accepts job messages one at a time in a continuous loop
#    - Predicts SCAM or SAFE with confidence %
#    - Highlights dangerous keywords found in the message
#    - Shows threat level alerts (LOW / MEDIUM / HIGH / CRITICAL)
#    - Keeps a live session scan counter and stats
#    - Responds to special commands: help, stats, history, clear, exit
# =============================================================

import os
import sys
import time
import random
import textwrap
from datetime import datetime

# ── Try importing model + preprocessor ───────────────────────
# Adjust paths if you keep files inside an agent/ subfolder
try:
    from model import FakeJobModel
    from preprocess import preprocess_text
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "agent"))
    from model import FakeJobModel
    from preprocess import preprocess_text


# ═══════════════════════════════════════════════════════════════
#  TERMINAL COLOR & STYLE CONSTANTS
# ═══════════════════════════════════════════════════════════════
# These are ANSI escape codes — special sequences that terminals
# interpret as color/style instructions instead of printing them.

# Text colors
BLACK   = "\033[30m"
RED     = "\033[91m"       # bright red  — danger / scam
GREEN   = "\033[92m"       # bright green — safe / ok
YELLOW  = "\033[93m"       # bright yellow — warning / medium risk
BLUE    = "\033[94m"       # bright blue
MAGENTA = "\033[95m"       # bright magenta
CYAN    = "\033[96m"       # bright cyan — system messages
WHITE   = "\033[97m"       # bright white

# Background colors
BG_RED    = "\033[41m"
BG_GREEN  = "\033[42m"
BG_YELLOW = "\033[43m"
BG_BLUE   = "\033[44m"
BG_BLACK  = "\033[40m"

# Text styles
BOLD      = "\033[1m"
DIM       = "\033[2m"
ITALIC    = "\033[3m"
UNDERLINE = "\033[4m"
BLINK     = "\033[5m"     # blinking text (supported on most terminals)
RESET     = "\033[0m"     # resets ALL styling back to terminal default


# ═══════════════════════════════════════════════════════════════
#  RISKY KEYWORD DATABASE
#  These are red-flag phrases commonly seen in Indian job scams.
#  Grouped by category for richer alert messages.
# ═══════════════════════════════════════════════════════════════

RISKY_KEYWORDS = {
    "💰 Money Trap": [
        "guaranteed", "guarantee", "fixed income", "daily income",
        "weekly income", "unlimited income", "earn daily", "earn weekly",
        "₹5000", "₹10000", "₹50000", "₹1 lakh", "lakh per month",
        "make money fast", "easy money", "big money", "high income",
        "passive income", "daily payout", "instant payout","urgent hiring", "work from home and earn", "no experience needed and earn", "earn from home", "home based job and earn","no inteview"
    ],
    "🔐 Fee / Deposit Scam": [
        "registration fee", "joining fee", "security deposit",
        "advance payment", "pay first", "send money", "transfer",
        "investment required", "upfront fee", "processing fee",
        "kit fee", "training fee", "deposit", "pay ₹", "bhejo",
    ],
    "🚨 No-Interview Red Flag": [
        "no interview", "without interview", "direct joining",
        "no experience needed", "no qualification", "anyone can join",
        "no exam", "no test", "koi bhi kar sakta", "bina interview",
    ],
    "📱 Suspicious Contact": [
        "whatsapp", "whatsapp karo", "telegram", "signal",
        "call now", "contact now", "abhi call", "message now",
        "send aadhaar", "send pan", "send photo", "share your details",
    ],
    "⏰ Urgency Pressure": [
        "urgent", "urgently", "limited seats", "last chance",
        "only today", "hurry", "act now", "jaldi karo", "abhi join",
        "offer expires", "closing soon", "don't miss", "mat chuko",
    ],
    "🏠 Work From Home Lure": [
        "work from home", "ghar se karo", "ghar baithe",
        "wfh", "home based", "घर से", "part time job",
        "2 ghante kaam", "3 ghante kaam", "flexible hours",
    ],
    "🎰 Too-Good-To-Be-True": [
        "100% genuine", "100% real", "100% safe", "guaranteed profit",
        "risk free", "risk-free", "no risk", "pakka income",
        "confirmed income", "assured", "double your money",
        "lottery", "lucky draw", "prize", "congratulations you won",
    ],
}

# Flatten all keywords into one list for quick scanning
ALL_RISKY_KEYWORDS = []
for cat_keywords in RISKY_KEYWORDS.values():
    ALL_RISKY_KEYWORDS.extend(cat_keywords)

# Model file path — adjust if you store model elsewhere
MODEL_PATH = "model/fake_job_model.pkl"


# ═══════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def typewrite(text, delay=0.018, newline=True):
    """
    Print text character by character for a dramatic terminal effect.

    sys.stdout.write() prints without adding a newline automatically.
    sys.stdout.flush() forces the output buffer to display immediately
    (by default, Python buffers terminal output and prints in chunks).

    Args:
        text  : The string to print
        delay : Seconds between each character (0.018 = ~55 chars/sec)
        newline: Whether to print a newline at the end
    """
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    if newline:
        print()


def slow_print(text, delay=0.008):
    """Faster typewrite for longer blocks of text."""
    typewrite(text, delay=delay)


def animate_scan(duration=1.2):
    """
    Shows a scanning animation bar while 'thinking'.

    \\r is a carriage return — moves cursor back to start of the line
    without moving down, so we can overwrite the same line repeatedly.
    This creates the animation effect.
    """
    frames = ["▓░░░░░░░░░", "▓▓░░░░░░░░", "▓▓▓░░░░░░░",
              "▓▓▓▓░░░░░░", "▓▓▓▓▓░░░░░", "▓▓▓▓▓▓░░░░",
              "▓▓▓▓▓▓▓░░░", "▓▓▓▓▓▓▓▓░░", "▓▓▓▓▓▓▓▓▓░",
              "▓▓▓▓▓▓▓▓▓▓"]
    start = time.time()
    i = 0
    while time.time() - start < duration:
        frame = frames[i % len(frames)]
        sys.stdout.write(f"\r  {CYAN}[SCANNING] {frame} {int((i/len(frames)%1)*100)+1}%{RESET}")
        sys.stdout.flush()
        time.sleep(0.09)
        i += 1
    sys.stdout.write(f"\r  {GREEN}[SCAN COMPLETE] ▓▓▓▓▓▓▓▓▓▓ 100%{RESET}          \n")
    sys.stdout.flush()


def scan_keywords(message: str) -> dict:
    """
    Scan a job message for risky keywords, organized by category.

    We check the lowercase version of the message against each keyword
    so matching is case-insensitive.

    Returns:
        dict: {category_name: [matched_keyword, ...]}
              Only includes categories that had at least one match.
    """
    msg_lower = message.lower()
    found = {}
    for category, keywords in RISKY_KEYWORDS.items():
        matches = [kw for kw in keywords if kw.lower() in msg_lower]
        if matches:
            found[category] = matches
    return found


def get_threat_level(scam_prob: float, keyword_count: int) -> tuple:
    """
    Determine threat level based on scam probability and keyword count.

    We combine both signals because:
    - High probability alone could be a borderline case
    - Many keywords with lower probability still means suspicious

    Returns:
        tuple: (level_string, color_code, alert_symbol)
    """
    score = scam_prob + (min(keyword_count, 10) * 0.02)  # keywords add up to 20% bonus

    if score >= 0.85:
        return "CRITICAL", RED,    "💀"
    elif score >= 0.65:
        return "HIGH",     RED,    "🚨"
    elif score >= 0.40:
        return "MEDIUM",   YELLOW, "⚠️ "
    else:
        return "LOW",      GREEN,  "✅"


def draw_confidence_bar(confidence: float, color: str, width: int = 30) -> str:
    """
    Draw a visual progress bar for confidence percentage.

    Example (90% confidence):
        [▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░] 90.0%

    Args:
        confidence: Float between 0 and 1
        color: ANSI color code string
        width: Total number of bar characters

    Returns:
        Formatted string with the bar + percentage
    """
    filled = int(confidence * width)
    empty  = width - filled
    bar    = f"{color}{'▓' * filled}{DIM}{'░' * empty}{RESET}"
    pct    = f"{color}{BOLD}{confidence*100:.1f}%{RESET}"
    return f"[{bar}] {pct}"


def wrap_text(text: str, width: int = 60, indent: str = "  ") -> str:
    """
    Word-wrap a long string with an indent prefix on each line.
    Useful for displaying long job messages neatly.
    """
    wrapped = textwrap.fill(text, width=width)
    return "\n".join(f"{indent}{line}" for line in wrapped.splitlines())


# ═══════════════════════════════════════════════════════════════
#  DISPLAY FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def print_banner():
    """
    Print the dramatic startup banner with ASCII art.
    Uses typewrite() for a boot-sequence feel.
    """
    os.system("cls" if os.name == "nt" else "clear")   # clear terminal screen

    banner = f"""
{CYAN}{BOLD}
 ██████╗ ███████╗████████╗███████╗ ██████╗████████╗
 ██╔══██╗██╔════╝╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝
 ██║  ██║█████╗     ██║   █████╗  ██║        ██║   
 ██║  ██║██╔══╝     ██║   ██╔══╝  ██║        ██║   
 ██████╔╝███████╗   ██║   ███████╗╚██████╗   ██║   
 ╚═════╝ ╚══════╝   ╚═╝   ╚══════╝ ╚═════╝   ╚═╝   
{RESET}{RED}{BOLD}
    ⚡  FAKE JOB DETECTION AGENT  ⚡
    🛡️  AI-Powered Cybersecurity System  🛡️
{RESET}{DIM}{CYAN}
    Version  : 2.0.0  |  Engine : TF-IDF + Logistic Regression
    Mode     : Active Threat Monitoring
    Coverage : Indian Job Market (Hinglish + English)
{RESET}"""
    print(banner)


def print_boot_sequence():
    """Simulated boot/initialization sequence for atmosphere."""
    boot_steps = [
        (f"{GREEN}[SYS]{RESET}  Initializing threat detection engine...", 0.4),
        (f"{GREEN}[SYS]{RESET}  Loading TF-IDF vocabulary matrix...",      0.3),
        (f"{GREEN}[SYS]{RESET}  Loading Logistic Regression classifier...", 0.3),
        (f"{GREEN}[SYS]{RESET}  Activating keyword threat scanner...",      0.3),
        (f"{GREEN}[SYS]{RESET}  Calibrating confidence scoring model...",   0.3),
        (f"{YELLOW}[SYS]{RESET} Loading Indian scam pattern database...",   0.4),
        (f"{YELLOW}[SYS]{RESET} Loading Hinglish linguistic patterns...",   0.3),
        (f"{GREEN}[SYS]{RESET}  All systems nominal. Agent is ONLINE.",     0.0),
    ]
    for step_text, pause in boot_steps:
        slow_print(f"  {step_text}")
        time.sleep(pause)
    print()


def print_result(message: str, prediction: int, scam_prob: float,
                 safe_prob: float, keywords_found: dict, scan_num: int):
    """
    Display the full analysis result for one job message.

    Args:
        message       : Original raw job text entered by user
        prediction    : 0 (SAFE) or 1 (SCAM)
        scam_prob     : Probability of SCAM (0.0 to 1.0)
        safe_prob     : Probability of SAFE (0.0 to 1.0)
        keywords_found: Dict of {category: [keywords]} from scan_keywords()
        scan_num      : Session scan number (for display)
    """
    total_kw = sum(len(v) for v in keywords_found.values())
    threat_level, threat_color, threat_icon = get_threat_level(scam_prob, total_kw)
    timestamp = datetime.now().strftime("%H:%M:%S")

    print(f"\n  {DIM}{'─'*58}{RESET}")
    print(f"  {DIM}SCAN #{scan_num:04d}  |  {timestamp}  |  THREAT LEVEL: "
          f"{threat_color}{BOLD}{threat_icon} {threat_level}{RESET}{DIM}  {RESET}")
    print(f"  {DIM}{'─'*58}{RESET}\n")

    # ── Message Preview ───────────────────────────────────────
    print(f"  {BOLD}{CYAN}📨 MESSAGE ANALYZED:{RESET}")
    print(wrap_text(message[:200] + ("..." if len(message) > 200 else ""), width=56))
    print()

    # ── VERDICT BOX ───────────────────────────────────────────
    if prediction == 1:  # SCAM
        print(f"  {BG_RED}{BLACK}{BOLD}{'':^54}{RESET}")
        print(f"  {BG_RED}{BLACK}{BOLD}{'  🚨  VERDICT: SCAM DETECTED  🚨':^54}{RESET}")
        print(f"  {BG_RED}{BLACK}{BOLD}{'':^54}{RESET}")
    else:  # SAFE
        print(f"  {BG_GREEN}{BLACK}{BOLD}{'':^54}{RESET}")
        print(f"  {BG_GREEN}{BLACK}{BOLD}{'  ✅  VERDICT: SAFE JOB POSTING  ✅':^54}{RESET}")
        print(f"  {BG_GREEN}{BLACK}{BOLD}{'':^54}{RESET}")

    # ── Confidence Bars ───────────────────────────────────────
    print()
    print(f"  {BOLD}📊 CONFIDENCE ANALYSIS:{RESET}")
    scam_bar = draw_confidence_bar(scam_prob, RED)
    safe_bar = draw_confidence_bar(safe_prob, GREEN)
    print(f"  {RED}SCAM{RESET} {scam_bar}")
    print(f"  {GREEN}SAFE{RESET} {safe_bar}")

    # ── Keyword Threat Report ─────────────────────────────────
    print()
    if keywords_found:
        print(f"  {BOLD}{YELLOW}🔍 THREAT INTELLIGENCE REPORT  [{total_kw} risks found]:{RESET}")
        for category, words in keywords_found.items():
            print(f"\n  {YELLOW}{BOLD}{category}{RESET}")
            # Show each risky keyword highlighted
            for kw in words:
                print(f"    {RED}▶{RESET} \"{RED}{BOLD}{kw}{RESET}\"")
    else:
        print(f"  {GREEN}🔍 KEYWORD SCAN: No suspicious patterns detected.{RESET}")

    # ── Threat-Level-Specific Alert Message ───────────────────
    print()
    if threat_level == "CRITICAL":
        alert_lines = [
            f"  {BG_RED}{BLACK}{BOLD} ⚠️  CRITICAL THREAT DETECTED ⚠️  {RESET}",
            f"  {RED}{BOLD}► This message shows MULTIPLE high-confidence scam signals.{RESET}",
            f"  {RED}► Do NOT respond, pay any fees, or share personal documents.{RESET}",
            f"  {RED}► Report this job posting to cybercrime.gov.in immediately.{RESET}",
            f"  {RED}► Block the sender's number/account right away.{RESET}",
        ]
    elif threat_level == "HIGH":
        alert_lines = [
            f"  {RED}{BOLD}⚠️  HIGH RISK — Strong scam indicators present.{RESET}",
            f"  {RED}► Never pay any fees or deposits for a job.{RESET}",
            f"  {RED}► Verify the company independently before responding.{RESET}",
            f"  {YELLOW}► Search the company name + 'scam' on Google.{RESET}",
        ]
    elif threat_level == "MEDIUM":
        alert_lines = [
            f"  {YELLOW}{BOLD}⚠️  MEDIUM RISK — Some suspicious patterns found.{RESET}",
            f"  {YELLOW}► Verify recruiter credentials before proceeding.{RESET}",
            f"  {YELLOW}► Never share Aadhaar/PAN upfront.{RESET}",
            f"  {YELLOW}► Legitimate companies don't ask for registration fees.{RESET}",
        ]
    else:
        if prediction == 0:
            alert_lines = [
                f"  {GREEN}{BOLD}✅  LOW RISK — This appears to be a genuine job posting.{RESET}",
                f"  {GREEN}► Still verify the company on LinkedIn or Naukri.com.{RESET}",
                f"  {DIM}► Trust your instincts. No job requires upfront payment.{RESET}",
            ]
        else:
            alert_lines = [
                f"  {YELLOW}{BOLD}⚠️  Possible scam — stay cautious.{RESET}",
                f"  {YELLOW}► Model confidence is moderate. Investigate further.{RESET}",
            ]

    for line in alert_lines:
        print(line)

    print(f"\n  {DIM}{'─'*58}{RESET}\n")


def print_help():
    """Display the available commands."""
    print(f"""
  {CYAN}{BOLD}╔══════════════════════════════════════════════╗
  ║           AGENT COMMAND REFERENCE            ║
  ╚══════════════════════════════════════════════╝{RESET}

  {BOLD}How to use:{RESET}
    Simply type or paste a job message and press Enter.
    The agent will scan it and return its verdict.

  {BOLD}Special Commands:{RESET}
    {CYAN}help{RESET}      →  Show this help screen
    {CYAN}stats{RESET}     →  Show session scan statistics
    {CYAN}history{RESET}   →  Show last 5 scanned messages
    {CYAN}clear{RESET}     →  Clear the terminal screen
    {CYAN}tip{RESET}       →  Show a random scam-awareness tip
    {CYAN}exit{RESET}      →  Shut down the agent

  {BOLD}Threat Levels:{RESET}
    {GREEN}✅ LOW      {RESET} Safe or very low risk
    {YELLOW}⚠️  MEDIUM   {RESET} Suspicious — verify before responding
    {RED}🚨 HIGH     {RESET} Strong scam signals — do not engage
    {RED}💀 CRITICAL {RESET} Confirmed scam pattern — block & report
""")


def print_stats(session: dict):
    """Display live statistics for the current session."""
    total   = session["total"]
    scams   = session["scams"]
    safes   = session["safes"]
    scam_pct = (scams / total * 100) if total else 0

    print(f"""
  {CYAN}{BOLD}╔══════════════════════════════════════════════╗
  ║            SESSION STATISTICS               ║
  ╚══════════════════════════════════════════════╝{RESET}

  {BOLD}Messages Scanned  :{RESET} {BOLD}{total}{RESET}
  {RED}Scams Detected    :{RESET} {RED}{BOLD}{scams}{RESET}
  {GREEN}Safe Jobs Found   :{RESET} {GREEN}{BOLD}{safes}{RESET}
  {YELLOW}Scam Rate         :{RESET} {YELLOW}{BOLD}{scam_pct:.1f}%{RESET}
  {DIM}Session Started   : {session['start_time']}{RESET}
""")


def print_history(session: dict):
    """Display last 5 scanned messages with verdicts."""
    history = session["history"]
    if not history:
        print(f"\n  {DIM}No scans yet in this session.{RESET}\n")
        return

    print(f"\n  {CYAN}{BOLD}Last {min(5, len(history))} Scans:{RESET}\n")
    for i, entry in enumerate(history[-5:][::-1], 1):
        tag   = f"{RED}SCAM{RESET}" if entry["prediction"] == 1 else f"{GREEN}SAFE{RESET}"
        short = entry["message"][:55] + "..." if len(entry["message"]) > 55 else entry["message"]
        conf  = entry["confidence"]
        print(f"  {DIM}{i}.{RESET} [{tag}] {conf:.0f}%  \"{short}\"")
    print()


TIPS = [
    "Legitimate companies NEVER ask for registration fees or security deposits.",
    "Any job promising ₹5000+ per day from home with no experience is almost certainly a scam.",
    "If you're contacted via WhatsApp or Telegram for a job — be very suspicious.",
    "Always search the company name + 'scam' on Google before applying.",
    "Government jobs are posted ONLY on official .gov.in websites — nowhere else.",
    "A real interview will NEVER ask for your Aadhaar or bank details upfront.",
    "Guaranteed income is a red flag. No real job can guarantee a fixed daily income.",
    "If the recruiter only communicates via WhatsApp and avoids phone calls — it's likely fake.",
    "Abroad job offers with upfront visa/agent fees are almost always scams.",
    "Check the recruiter's LinkedIn profile. Real recruiters have verifiable histories.",
    "MLM / network marketing jobs that require 'investing' money are pyramid schemes.",
    "Verify walk-in drives by calling the company's official number — not the one in the message.",
]


# ═══════════════════════════════════════════════════════════════
#  MAIN AGENT LOOP
# ═══════════════════════════════════════════════════════════════

def run_agent():
    """
    The main continuous monitoring loop.

    Flow:
        1. Print banner + boot sequence
        2. Load trained model (or train if not found)
        3. Enter infinite while loop:
            a. Show prompt
            b. Read user input
            c. Handle commands (help, stats, history, etc.)
            d. Preprocess + predict + display result
            e. Update session stats
        4. Exit gracefully on 'exit' or Ctrl+C
    """

    # ── Boot Display ──────────────────────────────────────────
    print_banner()
    time.sleep(0.5)
    print_boot_sequence()

    # ── Load Model ────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"  {YELLOW}[!] No trained model found at '{MODEL_PATH}'.{RESET}")
        print(f"  {YELLOW}[!] Please run: python train_model.py{RESET}\n")
        sys.exit(1)

    try:
        model = FakeJobModel.load(MODEL_PATH)
        print(f"  {GREEN}[✓] Threat detection model loaded successfully.{RESET}")
    except Exception as e:
        print(f"  {RED}[✗] Failed to load model: {e}{RESET}")
        sys.exit(1)

    # ── Session State ─────────────────────────────────────────
    # We track stats in a dict so we can show them with 'stats' command
    session = {
        "total"     : 0,
        "scams"     : 0,
        "safes"     : 0,
        "history"   : [],
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # ── Welcome Message ───────────────────────────────────────
    print()
    typewrite(f"  {GREEN}{BOLD}🛡️  Agent is ACTIVE. Monitoring mode engaged.{RESET}", delay=0.025)
    typewrite(f"  {CYAN}Type a job message to scan it, or type 'help' for commands.{RESET}", delay=0.020)
    print(f"  {DIM}{'─'*58}{RESET}\n")

    # ═══════════════════════════════════════════════════════════
    #  MAIN LOOP
    # ═══════════════════════════════════════════════════════════
    while True:
        try:
            # ── Prompt ────────────────────────────────────────
            # The prompt shows current scan count for awareness
            prompt = (f"{DIM}[#{session['total']+1:04d}]{RESET} "
                      f"{CYAN}{BOLD}AGENT>{RESET} ")
            user_input = input(prompt).strip()

            # ── Empty Input ───────────────────────────────────
            if not user_input:
                print(f"  {DIM}[Enter a job message or type 'help']{RESET}")
                continue

            cmd = user_input.lower()

            # ── Special Commands ──────────────────────────────

            if cmd in ("exit", "quit", "bye", "q"):
                # Graceful shutdown with final session summary
                print(f"\n  {CYAN}{BOLD}[AGENT] Shutting down...{RESET}")
                time.sleep(0.3)
                print_stats(session)
                typewrite(f"  {GREEN}🛡️  Stay safe out there. Agent offline.{RESET}", delay=0.025)
                print()
                break

            elif cmd == "help":
                print_help()
                continue

            elif cmd == "stats":
                print_stats(session)
                continue

            elif cmd == "history":
                print_history(session)
                continue

            elif cmd == "clear":
                os.system("cls" if os.name == "nt" else "clear")
                print_banner()
                continue

            elif cmd == "tip":
                tip = random.choice(TIPS)
                print(f"\n  {YELLOW}{BOLD}💡 SCAM AWARENESS TIP:{RESET}")
                print(f"  {YELLOW}{tip}{RESET}\n")
                continue

            # ── Scan the Job Message ──────────────────────────

            # Show scanning animation (makes it feel like work is happening)
            print()
            animate_scan(duration=random.uniform(0.8, 1.4))

            # Step 1: Preprocess the raw text
            clean_text = preprocess_text(user_input)

            # Step 2: Get prediction and probabilities from model
            #   predict()       → [0] or [1]  (the class label)
            #   predict_proba() → [[safe_prob, scam_prob]]
            prediction  = model.predict([clean_text])[0]
            proba       = model.predict_proba([clean_text])[0]
            safe_prob   = float(proba[0])   # probability it's SAFE
            scam_prob   = float(proba[1])   # probability it's SCAM

            trusted_companies = [
             "wipro", "tcs", "infosys", "google",
             "amazon", "microsoft", "accenture",
             "cognizant", "hcl", "ibm"
           ]
            
            safe_phrases = [
              "interview",
              "official",
              "recruitment",
              "walk-in",
              "campus drive",
              "software developer",
              "software engineer",
              "apply online"
            ]

            safe_found = any(word in user_input.lower() for word in safe_phrases)

            if safe_found:
              safe_prob += 0.20
              scam_prob -= 0.20

              # Keep probabilities valid
              scam_prob = max(0, min(scam_prob, 1))
              safe_prob = max(0, min(safe_prob, 1))

            trusted_found = any(company in user_input.lower() for company in trusted_companies)

            # Reduce scam probability slightly if trusted company found
            if trusted_found:
               scam_prob *= 0.7
               safe_prob = 1 - scam_prob

               if safe_prob > scam_prob:
                  prediction = 0

            # Step 3: Scan for risky keywords (independent of ML model)
            keywords_found = scan_keywords(user_input)

            # ── If model says SAFE but many keywords found: override ──
            # The keyword scanner is a safety net — if the ML model misses
            # something but multiple red-flag words are present, we bump
            # the scam probability up slightly.
            total_kw = sum(len(v) for v in keywords_found.values())
            if prediction == 0 and total_kw >= 4:
                # Soft override: average ML output with keyword signal
                keyword_signal = min(0.35 + total_kw * 0.05, 0.85)
                scam_prob = (scam_prob + keyword_signal) / 2
                safe_prob = 1 - scam_prob
                if scam_prob > 0.70:
                  prediction = 1
                else:
                  prediction = 0

            # Step 4: Update session counter
            session["total"] += 1
            if prediction == 1:
                session["scams"] += 1
            else:
                session["safes"] += 1

            # Step 5: Save to history
            confidence = scam_prob if prediction == 1 else safe_prob
            session["history"].append({
                "message"   : user_input,
                "prediction": prediction,
                "confidence": confidence * 100,
            })

            # Final prediction correction based on probabilities
            if safe_prob > scam_prob:
             prediction = 0
            else:
             prediction = 1

            # Step 6: Display full result
            print_result(
                message        = user_input,
                prediction     = prediction,
                scam_prob      = scam_prob,
                safe_prob      = safe_prob,
                keywords_found = keywords_found,
                scan_num       = session["total"],
            )

            # ── Running tally after every 5 scans ────────────
            if session["total"] % 5 == 0:
                print(f"  {DIM}📌 Session total: {session['total']} scanned | "
                      f"{session['scams']} scams | {session['safes']} safe{RESET}\n")

        except KeyboardInterrupt:
            # Ctrl+C — graceful exit
            print(f"\n\n  {YELLOW}[!] Interrupted by user (Ctrl+C).{RESET}")
            print_stats(session)
            typewrite(f"  {GREEN}🛡️  Agent offline. Stay vigilant!{RESET}", delay=0.020)
            print()
            break

        except Exception as e:
            # Don't crash the whole agent on one bad message
            print(f"\n  {RED}[ERROR] {e}{RESET}")
            print(f"  {DIM}Please try a different input.{RESET}\n")
            continue


# ── Entry Point ───────────────────────────────────────────────
if __name__ == "__main__":
    run_agent()