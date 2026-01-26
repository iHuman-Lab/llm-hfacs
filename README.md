# LLM-HFACS

## When Planes Have Bad Days, We Figure Out Why

> *"To err is human. To analyze those errors with LLMs is... this project."*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What Is This Sorcery?

Ever wondered what *really* causes aviation incidents? Spoiler: it's rarely just one thing. It's usually a spectacular domino effect of organizational chaos, supervisory slip-ups, human factors, and that one moment where someone thought "eh, what's the worst that could happen?"

**LLM-HFACS** is a data pipeline that takes raw aviation incident reports from NASA's ASRS (Aviation Safety Reporting System) and transforms them into structured insights using the **Human Factors Analysis and Classification System (HFACS)** framework — with a little help from our AI friends.

---

## The HFACS Pyramid of "How Did We Get Here?"

```
                    ┌─────────────────────────┐
                    │   ORGANIZATIONAL        │  ← "The fish rots from the head"
                    │   INFLUENCES (L4)       │     Resource issues, climate, processes
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   UNSAFE SUPERVISION    │  ← "My boss did what now?"
                    │         (L3)            │     Inadequate oversight, ignored problems
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   PRECONDITIONS FOR     │  ← "I haven't slept in 36 hours"
                    │   UNSAFE ACTS (L2)      │     Fatigue, stress, poor communication
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │     UNSAFE ACTS (L1)    │  ← "Oops"
                    │   Errors & Violations   │     The thing that actually happened
                    └─────────────────────────┘
```

---

## Features

- **Data Pipeline**: Ingests ASRS incident data and maps narratives to HFACS categories
- **Probability Analysis**: Computes conditional probabilities across the hierarchy (L4 → L3 → L2 → L1)
- **Full Chain Analysis**: Traces complete causal paths from organizational issues to unsafe acts
- **LLM Integration**: Supports both Ollama and OpenAI models for intelligent classification
- **Subcategory Deep Dives**: Granular analysis within each HFACS level
- **Metrics & Evaluation**: Precision, recall, F1 scores — because we're data scientists, not fortune tellers

---

## Installation

```bash
# Clone the repo
git clone https://github.com/elaheoveisi/llm-hfacs.git
cd llm-hfacs

# Create a virtual environment (trust us, you want this)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

```bash
# Run the main analysis pipeline
python src/main.py

# Run subcategory analysis
python src/main_subcategory.py
```

---

## Project Structure

```
llm-hfacs/
├── src/
│   ├── data/           # Data loading & HFACS mapping
│   ├── features/       # Probability computations & metrics
│   ├── models/         # LLM wrappers (Ollama/OpenAI)
│   ├── visualization/  # Making numbers pretty
│   ├── main.py         # Main pipeline
│   └── main_subcategory.py
├── configs/            # YAML configurations & HFACS mappings
├── prompts/            # LLM prompt templates (CoT, ToT, IO)
├── data/
│   ├── raw/            # Raw ASRS incident data
│   └── processed/      # Output analyses
└── reports/            # Generated reports
```

---

## How It Works

1. **Load** raw ASRS data (2015-2025 aviation incidents)
2. **Extract** factors from Anomaly, Human Factors, and Contributing Factors columns
3. **Map** factors to HFACS categories using predefined mappings
4. **Compute** conditional probabilities between levels
5. **Generate** full causal chains and probability matrices
6. **Profit** (in knowledge, not money — we're researchers)

---

## Tech Stack

| Category       | Tools                   |
| -------------- | ----------------------- |
| Data Wrangling | `pandas`, `openpyxl`    |
| LLM Framework  | `llama_index`           |
| LLM Providers  | Ollama, OpenAI          |
| Config         | `yaml`, `python-dotenv` |
| CLI            | `click`                 |
| Docs           | `Sphinx`                |

---

## Example Output

```
P(Inadequate_Supervision | Resource_Management) = 0.42
P(Condition_of_Operators | Inadequate_Supervision) = 0.67
P(Error | Condition_of_Operators) = 0.78

Full Chain: Resource_Management → Inadequate_Supervision → Condition_of_Operators → Error
Combined Probability: 0.22
```

*Translation: When organizations don't manage resources well, there's a 22% chance it cascades all the way down to an operational error. Fun!*

---

## Contributing

Found a bug? Have an idea? Want to add more levels to the pyramid of doom?

1. Fork it
2. Branch it (`git checkout -b feature/amazing-feature`)
3. Commit it (`git commit -m 'Add amazing feature'`)
4. Push it (`git push origin feature/amazing-feature`)
5. PR it

---

## Acknowledgments

- **NASA ASRS** for the incident data (and for making aviation safer)
- **HFACS Framework** for giving structure to chaos
- **Coffee** for making this possible

---

<p align="center">
  <i>Because every "oops" has a story, and that story has a spreadsheet.</i>
</p>

<p><small>Project based on the <a target="_blank" href="https://github.com/iHuman-Lab/ihuman-cookiecutter-data-science">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
