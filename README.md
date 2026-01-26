# 🚁 LLM-HFACS

> *Teaching AI to figure out why helicopters go "oops"*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**LLM-HFACS** is a research project that uses Large Language Models to automatically analyze aviation accident reports and classify them according to the **Human Factors Analysis and Classification System (HFACS)**. Because reading 215+ accident reports manually is *so* last decade.

---

## 🎯 What Does This Thing Do?

Imagine you're a safety analyst staring at a mountain of accident reports. Each one needs to be classified across 19 different human factors. Your coffee has gone cold. Your eyes are tired.

**Enter LLM-HFACS!** 🦸

This project takes those accident narratives and feeds them to LLMs that answer the eternal question: *"What went wrong and whose fault was it?"* (okay, we phrase it more scientifically than that)

```
📄 Accident Report  →  🤖 LLM Magic  →  ✅ HFACS Classification
```

---

## 🧠 The HFACS Framework

HFACS is a fancy taxonomy that breaks down human errors into 4 levels:

| Level                 | What It Means                  | Examples                                                 |
| --------------------- | ------------------------------ | -------------------------------------------------------- |
| **L1: Unsafe Acts**   | The pilot did a whoopsie       | Decision errors, skill-based errors, violations          |
| **L2: Preconditions** | The stage was set for disaster | Fatigue, poor communication, bad weather                 |
| **L3: Supervision**   | The boss dropped the ball      | Inadequate supervision, planned inappropriate operations |
| **L4: Organization**  | The system is broken           | Poor resource management, toxic organizational climate   |

Think of it as a blame pyramid 🔺 — the deeper you go, the more systemic the issue!

---

## 🤖 Supported Models

We've tested this with:

| Model                  | Where It Runs           | Vibe                                  |
| ---------------------- | ----------------------- | ------------------------------------- |
| `gpt-4o-mini`          | OpenAI Cloud ☁️          | Fast and cheap, our daily driver      |
| `qwen2.5:32b-instruct` | Ollama (local/remote) 🖥️ | When you want to keep your data close |

---

## 🎪 Prompting Strategies

We don't just ask the LLM once and call it a day. We've implemented **5 different prompting strategies** to see which one makes the AI think hardest:

| Strategy             | Description                     | Complexity |
| -------------------- | ------------------------------- | ---------- |
| **IO**               | Simple yes/no questions         | ⭐          |
| **IO Expanded**      | Detailed questions with context | ⭐⭐         |
| **IO Merged**        | All factors in one mega-prompt  | ⭐⭐         |
| **Chain-of-Thought** | "Think step by step..."         | ⭐⭐⭐        |
| **Tree-of-Thought**  | Hierarchical reasoning          | ⭐⭐⭐⭐       |

---

## 📁 Project Structure

```
llm-hfacs/
├── 📂 src/
│   ├── main.py              # 🧠 The brain - runs the whole show
│   ├── models/llm.py        # 🤖 LLM wrangling (OpenAI + Ollama)
│   ├── data/                # 📊 Data loading & cleaning
│   └── features/metrics.py  # 📈 Precision, Recall, F1 - oh my!
├── 📂 prompts/
│   ├── io.yaml              # Basic prompts
│   ├── io_expanded.yaml     # Detailed prompts
│   ├── cot.yaml             # Chain-of-thought prompts
│   └── tot.yaml             # Tree-of-thought prompts
├── 📂 data/
│   ├── data.json            # Raw accident reports (the good stuff)
│   └── results/             # Where the magic outputs live
└── 📂 configs/
    └── config.yaml          # API keys and settings
```

---

## 🚀 Getting Started

### 1. Clone & Install

```bash
git clone https://github.com/iHuman-Lab/llm-hfacs.git
cd llm-hfacs
pip install -r requirements.txt
```

### 2. Configure

Create/edit `configs/config.yaml` with your API keys:

```yaml
openai_api_key: "sk-your-key-here"
```

### 3. Run

```bash
python src/main.py
```

Then sit back and watch the progress bars go brrrrr 📊

---

## 📊 What You Get

After running, you'll find:

| Output                | What It Contains                 |
| --------------------- | -------------------------------- |
| `io_results.csv`      | LLM responses using IO prompting |
| `cot_results.csv`     | Chain-of-thought responses       |
| `tot_results.csv`     | Tree-of-thought responses        |
| `data/results/*.xlsx` | Precision, Recall, F1 scores     |
| Chi-squared stats     | LLM vs Human comparison          |

Plus a warm fuzzy feeling of automating tedious work ✨

---

## 🔬 Research Questions

This project investigates:

1. 🤔 Can LLMs match human experts in HFACS classification?
2. 📊 Which prompting strategy works best?
3. 🎯 How do different models compare?

**Spoiler:** Check the `data/results/` folder for answers!

---

## 📈 Metrics We Track

| Metric        | Question It Answers                            |
| ------------- | ---------------------------------------------- |
| **Precision** | When the AI says "YES", is it right?           |
| **Recall**    | Does the AI find all the factors humans found? |
| **F1 Score**  | The harmonious balance of both                 |


---

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Accident       │     │    Prompting    │     │      LLM        │
│  Reports        │────▶│    Strategy     │────▶│  (GPT/Qwen)     │
│  (JSON)         │     │  (IO/CoT/ToT)   │     │                 │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    Results      │◀────│    Metrics      │◀────│   YES/NO        │
│   (Excel/CSV)   │     │  Calculation    │     │   Responses     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

## 🤝 Contributing

Found a bug? Have an idea? PRs welcome!

Just remember: we're trying to make aviation *safer*, not give the AI ideas 😅

---

## 🏢 About

Built with ❤️ at **iHuman Lab**

*Making aviation safer, one LLM query at a time.*

<p><small>Project based on the <a target="_blank" href="https://github.com/iHuman-Lab/ihuman-cookiecutter-data-science">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

---

<p align="center">
  <i>"I asked GPT-4 to classify an accident report. It said 'skill-based error.' I felt personally attacked."</i>
  <br><br>
  🚁 Fly safe! 🚁
</p>
