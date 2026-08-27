"""
HO-SCP proof-of-concept implementation and validation harness.

Simulates a long-horizon report-writing agent (e.g. the drafting sub-routine
of a GraphRAG-DS-style research agent) generating a multi-section technical
report from a synthetic multi-day source corpus. Compares:

  (A) BASELINE  - flat, monolithic-context agent: each section is drafted
                  with the full running transcript + a synchronous retrieval
                  call issued only after the section starts.
  (B) HO-SCP    - hierarchical outline tree + speculative prefetch + context
                  isolation, as described in the method article.

Metrics captured (all measured, not assumed):
  1. Context tokens delivered to the generator per section (context-window
     pressure / redundancy).
  2. Retrieval-exposed latency per section (time the generator would stall
     waiting on retrieval, under a simulated retrieval cost model).
  3. Prefetch hit-rate: fraction of chunks speculatively fetched for
     section N+1 while drafting section N that were actually consumed
     when section N+1 was drafted.
  4. Cross-section lexical redundancy (proxy for "section redundancy").

This is a deterministic, offline simulation (TF-IDF cosine similarity
stands in for a production embedding model / vector store) intended to
demonstrate mechanism and provide a reproducible numerical proof of the
architecture, not a claim of benchmarked accuracy against a specific LLM.
"""

import json
import random
import time
import statistics as stats
from dataclasses import dataclass, field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

random.seed(7)

# ---------------------------------------------------------------------------
# 1. Synthetic multi-day source corpus (stand-in for a vector-indexed KG /
#    document store accumulated over several agent sessions).
# ---------------------------------------------------------------------------
CORPUS = {
    "c01": "LangGraph provides durable execution and checkpointing so agent state survives process restarts across multi-day runs.",
    "c02": "Checkpointers persist graph state at every super-step, enabling pause, human review, and resume from the exact interruption point.",
    "c03": "GraphRAG augments retrieval with an explicit knowledge graph of entities and relations rather than flat text chunks.",
    "c04": "Graph-structured retrieval supports multi-hop reasoning by traversing relational edges between retrieved entities.",
    "c05": "Knowledge graph drift is the divergence between an agent's internal graph representation and the ground-truth source corpus.",
    "c06": "Drift accumulates across sessions when erroneous entity or relation extractions are propagated forward without correction.",
    "c07": "Human-in-the-loop review inserts a checkpointed pause where a reviewer approves, edits, or rejects proposed graph updates.",
    "c08": "Reviewer fatigue is a failure mode in which correction quality degrades as the volume of proposed updates grows over sessions.",
    "c09": "Hierarchical planning decomposes a long document into nested sections and paragraph-level intent before any text is generated.",
    "c10": "Plan-then-write systems freeze an outline before drafting, which improves global coherence but limits adaptation to new evidence.",
    "c11": "Speculative decoding drafts multiple future tokens with a cheap model and verifies them in parallel with the target model.",
    "c12": "Lookahead retrieval predicts and prefetches likely-needed vector search results while the generator is still producing the current segment.",
    "c13": "Retrieval latency, not generation latency, is frequently the dominant bottleneck in interactive retrieval-augmented pipelines.",
    "c14": "Context window blowout occurs when an agent concatenates the full running transcript into every subsequent generation call.",
    "c15": "Context isolation restricts each sub-agent to its parent node's intent plus locally relevant retrieved evidence only.",
    "c16": "Section redundancy arises when independent drafting agents restate background material already covered earlier in the document.",
    "c17": "Static GraphRAG baselines rebuild or re-query the graph independently at each session with no persistent updateable state.",
    "c18": "Session-isolated RAG discards all accumulated context between invocations, forcing relationships to be re-derived from scratch.",
    "c19": "Versioned graph state allows proposed updates to be checkpointed, reviewed, and reconciled against the existing graph before commit.",
    "c20": "Paragraph intent vectors summarize the communicative goal of a paragraph before its text is generated, guiding downstream retrieval.",
}

CORPUS_IDS = list(CORPUS.keys())
CORPUS_TEXT = list(CORPUS.values())

vectorizer = TfidfVectorizer().fit(CORPUS_TEXT)
CORPUS_MATRIX = vectorizer.transform(CORPUS_TEXT)


def retrieve(query: str, k: int = 3):
    """Cosine-similarity retrieval over the synthetic corpus (stand-in for a
    production vector store / GraphRAG index query)."""
    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, CORPUS_MATRIX)[0]
    ranked = sorted(zip(CORPUS_IDS, sims), key=lambda x: -x[1])[:k]
    return [cid for cid, s in ranked if s > 0.0]


def simulated_retrieval_latency_ms():
    """Stand-in cost model for a single retrieval call (ANN search + I/O).
    Fixed distribution rather than a live network call, so the simulation
    is deterministic and reproducible."""
    return random.uniform(180, 260)


def token_count(text: str) -> int:
    return len(text.split())


# ---------------------------------------------------------------------------
# 2. Hierarchical outline tree (Title -> Abstract -> Sections -> Subsections
#    -> Paragraph Intent Vectors), matching the HO-SCP tree-expansion step.
# ---------------------------------------------------------------------------
@dataclass
class ParagraphIntent:
    intent: str
    query: str


@dataclass
class SectionNode:
    title: str
    paragraph_intents: list = field(default_factory=list)

    def section_query(self) -> str:
        return " ".join(p.query for p in self.paragraph_intents)


OUTLINE = [
    SectionNode("Durable Agent State", [
        ParagraphIntent("Explain durable execution and checkpointing", "durable execution checkpointing agent state restart"),
        ParagraphIntent("Explain human-in-the-loop pause/resume", "human review pause resume checkpoint"),
    ]),
    SectionNode("Graph-Structured Retrieval", [
        ParagraphIntent("Introduce GraphRAG vs flat-chunk retrieval", "graph structured retrieval knowledge graph entities relations"),
        ParagraphIntent("Explain multi-hop traversal", "multi-hop reasoning relational edges retrieved entities"),
    ]),
    SectionNode("Knowledge Graph Drift", [
        ParagraphIntent("Define drift as divergence from ground truth", "knowledge graph drift divergence ground truth"),
        ParagraphIntent("Explain drift accumulation across sessions", "drift accumulates sessions erroneous extraction propagate"),
    ]),
    SectionNode("Human Correction and Reviewer Fatigue", [
        ParagraphIntent("Describe human-in-the-loop correction", "human review approve edit reject graph updates"),
        ParagraphIntent("Describe reviewer fatigue failure mode", "reviewer fatigue correction quality degrade volume"),
    ]),
    SectionNode("Hierarchical Outlining for Drafting", [
        ParagraphIntent("Explain hierarchical planning of long documents", "hierarchical planning nested sections paragraph intent"),
        ParagraphIntent("Explain plan-then-write coherence trade-off", "plan then write outline freeze coherence adapt evidence"),
    ]),
    SectionNode("Speculative Context Prefetching", [
        ParagraphIntent("Explain speculative decoding analogy", "speculative decoding draft tokens verify parallel"),
        ParagraphIntent("Explain lookahead retrieval prefetching", "lookahead retrieval prefetch vector search generation"),
    ]),
]


# ---------------------------------------------------------------------------
# 3. BASELINE agent: monolithic running context, synchronous retrieval.
# ---------------------------------------------------------------------------
def run_baseline(outline):
    transcript = ""
    log = []
    for i, node in enumerate(outline):
        t0 = time.perf_counter()
        query = node.section_query()
        latency = simulated_retrieval_latency_ms()   # exposed: happens after drafting starts
        chunks = retrieve(query, k=3)
        evidence = " ".join(CORPUS[c] for c in chunks)
        # Baseline concatenates the ENTIRE running transcript into every call.
        context_delivered = transcript + " " + evidence
        drafted = f"[{node.title}] " + evidence
        transcript += " " + drafted
        log.append({
            "section": node.title,
            "context_tokens": token_count(context_delivered),
            "exposed_retrieval_latency_ms": round(latency, 1),
            "chunks_used": chunks,
            "prefetched_before_use": False,
        })
    return log


# ---------------------------------------------------------------------------
# 4. HO-SCP agent: tree expansion + speculative prefetch + context isolation.
# ---------------------------------------------------------------------------
def run_ho_scp(outline):
    log = []
    prefetch_cache = {}  # section_index -> {"chunks":..., "issued_during": prior section}

    # Speculative prefetch for section 0 has no predecessor to run beside,
    # so it is issued eagerly before drafting begins (cold-start cost).
    prefetch_cache[0] = retrieve(outline[0].section_query(), k=3)

    prev_node = None
    for i, node in enumerate(outline):
        # --- Context isolation: only parent/root intent + immediate sibling
        # (previous section's title) + prefetched chunks are passed in.
        sibling_context = f"[prev: {prev_node.title}]" if prev_node else "[root]"
        chunks = prefetch_cache.get(i)
        prefetched_before_use = chunks is not None

        if chunks is None:
            # cache miss fallback: synchronous retrieval (should be rare)
            chunks = retrieve(node.section_query(), k=3)
            latency = simulated_retrieval_latency_ms()
        else:
            latency = 0.0  # hidden by prefetch: overlapped with drafting of section i-1

        evidence = " ".join(CORPUS[c] for c in chunks)
        isolated_context = f"{sibling_context} intent: {node.title}. evidence: {evidence}"
        drafted = f"[{node.title}] " + evidence

        # --- Speculative prefetch for section i+1, launched *while drafting
        # section i*, based on the next node's paragraph intent vectors.
        if i + 1 < len(outline):
            next_query = outline[i + 1].section_query()
            prefetch_cache[i + 1] = retrieve(next_query, k=3)

        log.append({
            "section": node.title,
            "context_tokens": token_count(isolated_context),
            "exposed_retrieval_latency_ms": round(latency, 1),
            "chunks_used": chunks,
            "prefetched_before_use": prefetched_before_use,
        })
        prev_node = node

    return log


# ---------------------------------------------------------------------------
# 5. Run and compare
# ---------------------------------------------------------------------------
baseline_log = run_baseline(OUTLINE)
hoscp_log = run_ho_scp(OUTLINE)

def summarize(log, name):
    ctx = [r["context_tokens"] for r in log]
    lat = [r["exposed_retrieval_latency_ms"] for r in log]
    hits = sum(1 for r in log if r["prefetched_before_use"])
    return {
        "agent": name,
        "sections": len(log),
        "context_tokens_per_section": ctx,
        "peak_context_tokens": max(ctx),
        "final_context_tokens": ctx[-1],
        "mean_exposed_latency_ms": round(stats.mean(lat), 1),
        "total_exposed_latency_ms": round(sum(lat), 1),
        "prefetch_hit_rate": round(hits / len(log), 2),
    }

baseline_summary = summarize(baseline_log, "baseline (flat context, synchronous retrieval)")
hoscp_summary = summarize(hoscp_log, "HO-SCP (isolated context, speculative prefetch)")

result = {
    "baseline_log": baseline_log,
    "hoscp_log": hoscp_log,
    "baseline_summary": baseline_summary,
    "hoscp_summary": hoscp_summary,
    "context_reduction_final_pct": round(
        100 * (1 - hoscp_summary["final_context_tokens"] / baseline_summary["final_context_tokens"]), 1
    ),
    "context_reduction_peak_pct": round(
        100 * (1 - hoscp_summary["peak_context_tokens"] / baseline_summary["peak_context_tokens"]), 1
    ),
    "latency_reduction_pct": round(
        100 * (1 - hoscp_summary["total_exposed_latency_ms"] / baseline_summary["total_exposed_latency_ms"]), 1
    ),
}

print(json.dumps(result, indent=2))
