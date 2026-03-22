# AI Co-worker Engine

An AI workplace-simulation platform built with Streamlit, LangGraph, and Ollama. The system simulates a cross-functional group of AI co-workers that respond as role-based stakeholders, route requests through a hidden supervisor, retrieve scenario knowledge, execute business-style tools, and export polished artifacts.

This repository implements a rollout simulation with three key personas:

- CEO
- CHRO
- Employer Branding & Internal Communications Regional Manager

## Demo Flow

1. The user asks a question in the Streamlit chat UI.
2. A safety node checks for forbidden language.
3. The Supervisor routes the request to a specific persona or initiates a meeting flow.
4. The selected persona executes by reading its `SOUL.md`, session `Knowledge.md`, conversation history, and retrieved knowledge.
5. The persona dynamically calls external tools if required.
6. The final response is delivered to the chat.
7. The user can capture the response as an artifact (e.g., Final Plan, Internal Communication, Executive Update).
8. Once all required artifacts are collected, the portfolio pack can be exported as a PDF.

### Core Modules

- `app.py`: Streamlit UI, chat loop, sidebar, portfolio actions.
- `my-app/coworker_engine/engine.py`: LangGraph graph definition and routing logic.
- `my-app/coworker_engine/agent.py`: Hidden Supervisor planning node.
- `my-app/coworker_engine/utils/nodes.py`: Persona nodes, safety handling, meeting synthesis.
- `my-app/coworker_engine/simulation.py`: Scenario and persona configurations.
- `my-app/coworker_engine/utils/knowledge.py`: Retrieval corpus and similarity search.
- `my-app/coworker_engine/utils/agent_memory.py`: File-backed memory management.
- `my-app/coworker_engine/utils/tools.py`: Tool surface for integrations.
- `my-app/coworker_engine/utils/portfolio.py`: Artifact registry and PDF export.
- `my-app/coworker_engine/utils/safety.py`: Forbidden language guardrails.

### Personas

The active simulation defines three personas with distinct identities, system prompts, and reputation triggers. Users can directly tag them using aliases, for example:
- **CEO**: `@executive`, `@ceo`, `@leadership`
- **CHRO**: `@people`, `@talent`, `@hr`, `@chro`
- **Employer Branding & Internal Communications Regional Manager**: `@operations`, `@ops`, `@regional`

### Supervisor

The Supervisor agent operates transparently in the background to determine whether the user's intent warrants a direct response from one coworker or a synthesized multi-stakeholder meeting. It also provides automatic hints if the user appears stuck.

### Mock Jira Integration
Personas have access to operational context via tool execution. The application provides a bundled Fake Jira Flask service under `my-app/fake_jira` which runs by default on `http://127.0.0.1:5000`. Available tools include Jira task listing, creation, commenting, and status updates.

## Portfolio Pack Export

Users can save assistant outputs as structured artifacts:
- `final_plan`
- `internal_comm`
- `executive_update`

When readiness criteria are met, the artifacts are compiled, enriched with source notes from retrieved knowledge, and generated as a ReportLab PDF under `exports/portfolio-packs/<thread_id>/`.

## Project Structure

```text
.
├── agent_memory/
│   ├── executive/
│   ├── operations/
│   ├── people/
│   ├── sessions/
│   └── supervisor/
├── app.py
├── doc/
│   ├── daily-distill-feature-plan.md
│   └── simulation-modules/
│       └── module-2.1-script.md
├── exports/
├── langgraph.json
├── my-app/
│   ├── coworker_engine/
│   │   ├── agent.py
│   │   ├── engine.py
│   │   ├── simulation.py
│   │   └── utils/
│   ├── fake_jira/
│   └── requirements.txt
├── ollama/
│   └── Modelfile
├── pyproject.toml
├── scripts/
│   ├── evaluate_engine.py
│   ├── portfolio_smoke.py
│   ├── run_demo.py
│   ├── run_fake_jira.py
│   └── setup_ollama_model.sh
└── tests/
```

## Requirements
- Python 3.9+
- One model provider:
  - Ollama (installed and running locally), or
  - OpenAI API access
- Python packages specified in `pyproject.toml`

## Installation

```bash
git clone https://github.com/xuanquynhphamvu/edtronaunt-ai-co-worker.git
cd edtronaunt-ai-co-worker

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

## Configuration

Create a `.env` file in the repository root:
```bash
MODEL_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:32b
FAKE_JIRA_BASE_URL=http://127.0.0.1:5000
```
To use OpenAI instead of Ollama:
```bash
MODEL_PROVIDER=openai
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-4o-mini
FAKE_JIRA_BASE_URL=http://127.0.0.1:5000
```
Pull the required model:
```bash
bash scripts/setup_ollama_model.sh 
# OR manually: ollama pull qwen2.5:32b
```

## Running the Application

**Quick Start (Demo Path):**
Runs both the Streamlit UI and the Mock Jira backend.
```bash
python scripts/run_demo.py
```

**Running Services Manually:**
```bash
# Terminal 1: run mock backend
python scripts/run_fake_jira.py

# Terminal 2: run Streamlit UI
streamlit run app.py
```

## Example Prompts

**Direct-to-persona:**
- `@executive What should leadership prioritize if this rollout risks spreading the team too thin?`
- `@people How do we improve adoption without adding too much training overhead?`

**Cross-functional:**
- `Design a rollout plan that balances strategy clarity, people adoption, and operational realism.`

**Tool hooks:**
- `@operations Show me the current Jira tasks.`
- `@people Add a comment to task 12 saying managers need enablement before the pilot.`

**Artifact capture:**
- `Write a concise executive update for leadership.`

## System Limitations & Roadmap

**Current Limitations:**
- Retrieval uses lightweight token vectors rather than semantic embeddings.
- Durable conversation state across LangGraph invocations resets on restart (no checkpointer implemented yet).
- Knowledge scope expands per session but currently lacks periodic pruning or summarization.

**Upcoming Enhancements:**
1. Complete daily distillation functionality to sync session knowledge into shared persona `SOUL.md` baselines.
2. Upgrade retrieval to use a true local embedding model.
3. Integrate LangGraph memory checkpointers for long-term state routing.
4. Abstract simulations deeper to allow runtime scenario switching from the UI.
5. Bolster offline evaluation layers with automated snapshot checks on portfolio artifacts.

## Testing
Run the project's test suite to validate memory schemas, routing functionality, and artifact isolation:
```bash
python -m unittest tests.test_agent_memory tests.test_visible_responses tests.test_portfolio tests.test_fake_jira
```
