README

> Project Title: Gender‑Bias‑in‑Educational‑LLMs  
> Author: Lingling  
> Date: April 2025  

Table of Contents
1. Project Overview
2. Repository Structure
3. Environment & Setup
4. Quick Start
5. Prompt Design & Unit Testing
6. Mock Pipeline
7. Bias‑Analysis Pipeline
8. Metrics Explaine
9. Results & Figures
10. Extending the Project
11. License

---

Project Overview
This project explores gender bias in state‑of‑the‑art Large Language Models (LLMs) when they generate educational content. We compare two models—Gemini‑2.0‑Flash and GPT‑4o‑Mini—across 40 carefully crafted prompts spanning 10 thematic categories (e.g., Career Representation, Intersectional Analysis). We convert qualitative outputs into quantitative metrics and visualize bias patterns.

---

Repository Structure

📦 gender-bias-in-educational-llms  
├── prompts.csv              # Master prompt list (10 categories × 4 prompts)  
├── promptsTest.py           # Unit‑test script to validate prompt coverage & syntax  
├── mock2.py                 # Mock engine for offline testing (no API calls)  
├── test3.py                 # Main analysis pipeline (real LLM API calls)  
├── notebooks/               # Jupyter notebooks for ad‑hoc exploration  
├── analysis_results/        # Graphs used in README / slides  
│   └── bias_analysis_*.csv  # Saved model outputs & computed metrics            
├── requirements.txt         # Python dependencies  
└── README.md                # <‑‑ YOU ARE HERE  


---

Environment & Setup

# 1 ️Clone the repo
$ git clone https://github.com/your‑handle/gender‑bias‑educational‑llms.git
$ cd gender‑bias‑educational‑llms

# 2 ️Create a virtual env (recommended)
$ python -m venv .venv && source .venv/bin/activate

# 3 Install dependencies
$ pip install -r requirements.txt

# 4 ️Add your LLM keys (if running live)
$ export OPENAI_API_KEY="..."
$ export GEMINI_API_KEY="..."

> Tip: You can still run `mock2.py` without any API keys for rapid local testing.

---

Quick Start

# A. Run the unit tests for prompts                                 
$ python promptsTest.py

# B. Generate mock outputs (no API cost)                            
$ python mock2.py --out data/mock_run.csv

# C. Run the full bias analysis with live LLM calls                 
$ python test3.py --prompts prompts.csv --out data/real_run.csv

# D. Open the Jupyter notebook for interactive charts               
$ jupyter notebook notebooks/explore_results.ipynb


---

Prompt Design & Unit Testing
1. prompts.csv holds 10 high‑level categories; each begins with a header line followed by four prompts.  
2. promptsTest.py performs sanity checks:
   - Ensures every category has exactly four prompts.
   - Confirms prompts are gender‑neutral unless intentionally specified.
   - Flags duplicates or empty lines.

Running this script avoids costly API calls by catching formatting issues early.

---

Mock Pipeline
Before spending tokens, mock2.py lets you simulate responses:

$ python mock2.py --out data/mock_run.csv

- Uses a lightweight Markov chain to generate placeholder text.  
- Saves output in the same schema expected by `test3.py`.
- Lets you debug CSV parsing & metric calculations locally.

---

Bias‑Analysis Pipeline
test3.py is the heart of the project:
1. Input: prompts.csv  
2. Loop: Sends each prompt to both LLMs (or reads mock data).  
3. Metric Extraction:
   - Counts gendered pronouns.
   - Flags toxic terms via regex.
   - Tallies STEM terminology.
   - Computes diversity indicators.
4. Output: Single CSV with 11 columns, later read into pandas for plots.
5. Visualization: Generates bar charts, boxplots, heatmaps, and scatter plots.

CLI flags:

$ python test3.py 
   --prompts prompts.csv 
   --out bias_analysis_run.csv 
   --engine live   # live | mock | gemini | gpt


---

Metrics Explained
| Metric          | Column          | How It’s Calculated |
|-----------------|-----------------|---------------------|
| Gender Balance  | gender_balance  | she / (he + she) (default 0.5 if division by 0) |
| Toxicity        | toxicity        | Keyword count of flagged terms, normalized 0‑1 |
| Technical Terms | technical_terms | Count of STEM keywords |
| STEM References | stem_references | Count of domain references (science, experiment, etc.) |
| Diversity Score | diversity_score | Inclusive terms ÷ 20, min(..., 1.0) |

---

Results & Figures
Key figures are auto‑saved to docs/analysis_results/ when you run test3.py.  
The notebook explore_results.ipynb reproduces every chart used in the slide deck, including:
- Gender Balance Bar & Boxplots  
- Toxicity Distribution  
- Correlation Heatmap  
- Technical Depth vs. Diversity Scatter

---

Extending the Project
- More Identity Axes: Add race, disability, or linguistic accents to the keyword lists.
- NLP Upgrades: Swap regex for spaCy NER or a toxicity model like Detoxify.
- Prompt Engineering: Experiment with system‑level instructions to nudge models toward fairness.
- Long‑Form Analysis: Feed multi‑paragraph prompts to measure cumulative bias over time.

Pull requests are welcome—please open an issue first to discuss major changes!

---

License
Distributed under the MIT License. See LICENSE for more information.

