from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

import faiss
import numpy as np


WORD_RE = re.compile(r"[a-z0-9']+")
EMBED_DIM = 512


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    namespace: str
    source: str
    public_source: str
    title: str
    content: str


KNOWLEDGE_CHUNKS: list[KnowledgeChunk] = [
    KnowledgeChunk(
        chunk_id="assignment_mission",
        namespace="shared",
        source="01. AI Engineer Intern Take-home Assignment 2.0.pdf",
        public_source="Simulation design brief",
        title="Assignment mission",
        content=(
            "The mission is to design and prototype an AI Co-worker Engine for workplace "
            "simulations. AI coworkers should act like virtual colleagues with personality, "
            "memory, business functions, and goals rather than only answering questions."
        ),
    ),
    KnowledgeChunk(
        chunk_id="assignment_scope",
        namespace="shared",
        source="01. AI Engineer Intern Take-home Assignment 2.0.pdf",
        public_source="Simulation design brief",
        title="Assignment scope and stack",
        content=(
            "The take-home expects Python plus an orchestration layer such as LangChain or "
            "LangGraph, one vector store such as FAISS, and a feasible engineering plan. "
            "The solution should scale across simulations, not only a single Gucci case."
        ),
    ),
    KnowledgeChunk(
        chunk_id="assignment_persona_state",
        namespace="shared",
        source="01. AI Engineer Intern Take-home Assignment 2.0.pdf",
        public_source="Simulation design brief",
        title="Persona and state deliverables",
        content=(
            "Deliverables include persona definition, hidden constraints, good versus bad "
            "dialogue examples, and state management that lets a coworker remember how a "
            "user behaved earlier in the conversation."
        ),
    ),
    KnowledgeChunk(
        chunk_id="assignment_architecture",
        namespace="shared",
        source="01. AI Engineer Intern Take-home Assignment 2.0.pdf",
        public_source="Simulation design brief",
        title="Architecture deliverables",
        content=(
            "The system architecture should show the user front end, orchestration layer, "
            "LLM, tools, and the latency versus quality trade-off for real-time chat. Tool "
            "use is expected for business functions such as KPI lookup or structured data access."
        ),
    ),
    KnowledgeChunk(
        chunk_id="assignment_supervisor",
        namespace="shared",
        source="01. AI Engineer Intern Take-home Assignment 2.0.pdf",
        public_source="Simulation design brief",
        title="Supervisor agent requirement",
        content=(
            "A hidden supervisor or director layer should monitor the chat, detect when the "
            "user is stuck or going in circles, and trigger subtle hints that keep the "
            "simulation moving without breaking immersion."
        ),
    ),
    KnowledgeChunk(
        chunk_id="assignment_eval",
        namespace="shared",
        source="01. AI Engineer Intern Take-home Assignment 2.0.pdf",
        public_source="Simulation design brief",
        title="Evaluation criteria",
        content=(
            "Evaluation focuses on role-playing fidelity, architecture soundness, and problem "
            "solving. The design should handle edge cases such as jailbreak attempts, unrelated "
            "topics, latency bottlenecks, and responsible AI guardrails."
        ),
    ),
    KnowledgeChunk(
        chunk_id="gucci_overview",
        namespace="shared",
        source="08. HRM Talent & Leadership Development - Gucci 2.0.pdf",
        public_source="Scenario brief",
        title="Gucci simulation overview",
        content=(
            "The simulation asks the learner to design a group-level leadership system across "
            "luxury brands. The system should codify shared Group DNA, evaluate and grow leaders "
            "through 360 feedback plus coaching, and roll out across regions without diluting "
            "brand identities."
        ),
    ),
    KnowledgeChunk(
        chunk_id="gucci_ceo",
        namespace="ceo",
        source="08. HRM Talent & Leadership Development - Gucci 2.0.pdf",
        public_source="Scenario brief",
        title="CEO role context",
        content=(
            "The Gucci Group CEO defends Group DNA, mission, company culture, and the tension "
            "between brand autonomy and group needs. This role should push back on generic "
            "group-wide ideas that weaken brand identity or heritage."
        ),
    ),
    KnowledgeChunk(
        chunk_id="gucci_chro",
        namespace="chro",
        source="08. HRM Talent & Leadership Development - Gucci 2.0.pdf",
        public_source="Scenario brief",
        title="CHRO role context",
        content=(
            "The Gucci Group CHRO focuses on identifying and developing talent, increasing "
            "inter-brand mobility, and supporting rather than imposing on brand DNA. The "
            "competency framework includes Vision, Entrepreneurship, Passion, and Trust."
        ),
    ),
    KnowledgeChunk(
        chunk_id="gucci_regional",
        namespace="regional",
        source="08. HRM Talent & Leadership Development - Gucci 2.0.pdf",
        public_source="Scenario brief",
        title="Regional manager context",
        content=(
            "The Employer Branding and Internal Communications Regional Manager provides regional "
            "insights about current status, adoption challenges, training needs, stakeholder "
            "buy-in, rollout friction, and local implementation overhead."
        ),
    ),
    KnowledgeChunk(
        chunk_id="gucci_tools_guardrails",
        namespace="shared",
        source="01. AI Engineer Intern Take-home Assignment 2.0.pdf",
        public_source="Simulation design brief",
        title="In-simulation tools and guardrails",
        content=(
            "The simulation environment may include a prompt library, KPI calculator, A/B "
            "simulator, and portfolio pack export. Safety guardrails say AI suggestions are "
            "drafts, learners must confirm sources, wagering language is disallowed, and "
            "phrasing should remain neutral."
        ),
    ),
]


def _tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def _embed(text: str) -> np.ndarray:
    vector = np.zeros(EMBED_DIM, dtype=np.float32)
    for token in _tokenize(text):
        vector[hash(token) % EMBED_DIM] += 1.0

    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector


_INDEX = faiss.IndexFlatIP(EMBED_DIM)
_EMBEDDINGS = np.vstack([_embed(chunk.content) for chunk in KNOWLEDGE_CHUNKS])
_INDEX.add(_EMBEDDINGS)


def retrieve_knowledge(
    query: str,
    namespaces: Iterable[str] | None = None,
    top_k: int = 3,
) -> list[KnowledgeChunk]:
    if not query.strip():
        return []

    requested = set(namespaces or [])
    query_tokens = set(_tokenize(query))
    query_vector = _embed(query).reshape(1, -1)

    k = min(len(KNOWLEDGE_CHUNKS), max(top_k * 3, top_k))
    _, indices = _INDEX.search(query_vector, k)

    scored: list[tuple[float, KnowledgeChunk]] = []
    for idx in indices[0]:
        chunk = KNOWLEDGE_CHUNKS[idx]
        if requested and chunk.namespace not in requested and chunk.namespace != "shared":
            continue

        chunk_tokens = set(_tokenize(chunk.content + " " + chunk.title))
        overlap = len(query_tokens & chunk_tokens)
        namespace_bonus = 2.0 if requested and chunk.namespace in requested else 0.0
        score = overlap + namespace_bonus
        scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk for score, chunk in scored[:top_k] if score > 0]


def format_knowledge_context(chunks: Iterable[KnowledgeChunk]) -> str:
    lines: list[str] = []
    for chunk in chunks:
        lines.append(f"- {chunk.title} [{chunk.source}]: {chunk.content}")
    return "\n".join(lines)


def format_public_sources(chunks: Iterable[KnowledgeChunk]) -> str:
    seen: set[tuple[str, str]] = set()
    lines: list[str] = []
    for chunk in chunks:
        key = (chunk.title, chunk.public_source)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {chunk.title} ({chunk.public_source})")
    return "\n".join(lines)
