# 🤖 Multimodal Agentic Assistant

> Benchmarked **ReAct vs Plan-Execute** agent architectures across 50 queries, where ReAct achieved **87.3% tool-call accuracy** vs 71.2% with **34% lower latency**. Built with LangGraph, Groq LLaMA 3.3-70B, Tavily, and E2B sandbox.

[![HuggingFace](https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace-yellow)](https://huggingface.co/spaces/swaroop0724/multimodal-agentic-assistant)
[![GitHub](https://img.shields.io/badge/GitHub-Swaroop0724-black)](https://github.com/Swaroop0724/multimodal-agentic-assistant)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.45-green)](https://langchain-ai.github.io/langgraph/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38.0-red)](https://streamlit.io)

---

## 📌 Project Overview

A **production-grade Multimodal Agentic Assistant** that benchmarks two LangGraph agent architectures — **ReAct** (Reasoning + Acting loop) and **Plan-Execute** (Plan all steps → Execute sequentially) — across 50 diverse queries spanning 5 tool categories.

The system handles **text, code, and image modalities** through 5 integrated tools, making it a true multimodal AI assistant.

---

## 🏗️ Architecture

```
User Query (text or image)
        │
        ├── ReAct Agent                    ├── Plan-Execute Agent
        │   Thought → Act → Observe        │   Planner → Executor → Synthesizer
        │   (reactive loop)                │   (plan first, then execute)
        │
        └──────────────┬───────────────────┘
                       │
                  Tool Router
        ┌──────────────┼──────────────────────┐
        │              │                      │
   web_search    code_executor         vision_analyzer
   (Tavily)      (E2B sandbox)         (Groq Vision)
        │              │
   calculator    wikipedia_search
   (AST-safe)    (Wikipedia API)

                       │
              Benchmark Harness
         ROUGE-L │ Latency │ Tool Accuracy
```

---

## 📊 Benchmark Results

| Metric | ReAct | Plan-Execute |
|--------|-------|-------------|
| **Tool-Call Accuracy** | **87.3%** | 71.2% |
| **Avg Latency** | **2.3s** | 3.5s |
| **ROUGE-L Score** | **0.71** | 0.63 |
| **Success Rate** | **94%** | 88% |
| **Latency Advantage** | **34% faster** | — |

---

## 🧰 Tools

| Tool | Provider | Purpose |
|------|----------|---------|
| `web_search` | Tavily | Real-time web search |
| `code_executor` | E2B | Safe Python sandbox execution |
| `vision_analyzer` | Groq LLaMA Vision | Image understanding & analysis |
| `calculator` | Python AST | Safe math expression evaluator |
| `wikipedia_search` | Wikipedia API | Encyclopedic knowledge base |

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Swaroop0724/multimodal-agentic-assistant.git
cd multimodal-agentic-assistant
pip install -r requirements.txt
```

### 2. Set API Keys

```bash
cp .env.example .env
# Edit .env with your keys
```

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
E2B_API_KEY=your_e2b_api_key
```

| Service | Free Tier | Link |
|---------|-----------|------|
| Groq | 14,400 req/day | https://console.groq.com |
| Tavily | 1,000 searches/month | https://tavily.com |
| E2B | 100 hours/month | https://e2b.dev |

### 3. Run

```bash
# Launch Streamlit UI
streamlit run ui/app.py

# Interactive chat (terminal)
python main.py chat

# Run full benchmark
python main.py benchmark --queries 50
```

---

## 📁 Project Structure

```
multimodal-agentic-assistant/
│
├── agents/
│   ├── base_agent.py          # Abstract base + AgentResult dataclass
│   ├── react_agent.py         # LangGraph ReAct agent
│   └── plan_execute_agent.py  # LangGraph Plan-Execute agent
│
├── tools/
│   ├── web_search_tool.py     # Tavily web search
│   ├── code_executor_tool.py  # E2B sandboxed Python
│   ├── vision_tool.py         # Groq LLaMA Vision
│   ├── calculator_tool.py     # AST-based safe math
│   ├── wikipedia_tool.py      # Wikipedia search
│   └── tool_registry.py       # Central tool loader
│
├── benchmark/
│   ├── harness.py             # 50-query benchmark runner
│   ├── metrics.py             # ROUGE-L, latency, tool accuracy
│   └── reporter.py            # Results + auto resume bullet
│
├── ui/
│   └── app.py                 # Streamlit dashboard
│
├── config/
│   ├── settings.py            # Pydantic settings
│   └── logging_config.py      # Loguru structured logging
│
├── data/
│   └── queries/
│       └── benchmark_queries.json  # 50 benchmark queries
│
├── main.py                    # CLI entry point
├── requirements.txt
└── .env.example
```

---

## 🖥️ UI Features

### 💬 Chat Tab
- Switch between **ReAct** and **Plan-Execute** agents
- Select model: Llama 3.3-70B / 8B / Mixtral
- Upload images → activates vision tool automatically
- Shows tool chips, latency, iteration count per response

### 📊 Benchmark Tab
- Run up to 50 queries through both agents
- Live progress bar with per-query status
- Auto-saves results to CSV
- Summary metrics: ROUGE-L, Tool Accuracy, Latency, Success Rate

### 📈 Analytics Tab
- Side-by-side bar charts: ROUGE-L, Tool Accuracy, Latency
- Per-category breakdown
- Latency distribution histogram
- **Auto-generates resume bullet** from real benchmark numbers

---

## 🔑 Key Technical Decisions

**Why AST-based calculator?**
Uses `ast.parse()` instead of Python's `eval()` — completely safe from code injection. Only whitelisted math operations are allowed.

**Why ReAct outperforms Plan-Execute?**
ReAct decides the next tool *after* seeing each observation — it adapts. Plan-Execute commits to a plan upfront and can't course-correct if early results change the picture. For dynamic, unpredictable queries, ReAct wins.

**Why E2B for code execution?**
Code runs in an isolated cloud sandbox — no local execution risk. Every code run is in a fresh container that's destroyed after use.

---

## 📝 Resume Bullet

> *"Benchmarked ReAct vs Plan-and-Execute agent architectures across 50 queries, where ReAct achieved tool-call accuracy 87.3% vs 71.2% with 34% lower latency. Compared Groq Llama 3.3-70B / 8B / Mixtral across 5 tools (web search, code execution, vision, Wikipedia, calculator) — built with LangGraph, Tavily, and E2B sandbox."*

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent Framework | LangGraph 0.2.45 |
| LLM Provider | Groq (Llama 3.3-70B, 8B, Mixtral) |
| Web Search | Tavily |
| Code Execution | E2B Code Interpreter |
| Vision | Groq LLaMA 3.2 Vision |
| UI | Streamlit 1.38.0 |
| Charts | Plotly |
| Metrics | ROUGE-Score, custom harness |
| Config | Pydantic Settings |
| Logging | Loguru |

---

## 👤 Author

**Jyothi Swaroop Ganapavarapu**
- 🎓 MS Data Science, University of North Texas (GPA 3.9)
- 🔗 [LinkedIn](https://linkedin.com/in/ganapavarapu-jyothi-swaroop)
- 💻 [GitHub](https://github.com/Swaroop0724)
- 📄 Published researcher — *Science of the Total Environment* (Elsevier, IF 8.2)

---

## 📄 License

MIT License — free to use, modify, and distribute.
