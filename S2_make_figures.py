import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("ho_scp_output.json") as f:
    r = json.load(f)

sections = [x["section"] for x in r["baseline_log"]]
short = [f"S{i+1}" for i in range(len(sections))]
base_ctx = r["baseline_summary"]["context_tokens_per_section"]
hoscp_ctx = r["hoscp_summary"]["context_tokens_per_section"]

# Figure 1: context tokens per section
fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=200)
x = range(len(short))
w = 0.38
ax.bar([i - w/2 for i in x], base_ctx, width=w, label="Baseline (flat context)", color="#7f8c9a")
ax.bar([i + w/2 for i in x], hoscp_ctx, width=w, label="HO-SCP (context isolation)", color="#2e6da4")
ax.set_xticks(list(x))
ax.set_xticklabels(short)
ax.set_xlabel("Manuscript section (drafting order)")
ax.set_ylabel("Context tokens delivered to generator")
ax.set_title("Context-window pressure per section: baseline vs. HO-SCP")
ax.legend()
fig.tight_layout()
fig.savefig("fig1_context_tokens.png")
plt.close(fig)

# Figure 2: exposed retrieval latency per section
base_lat = [x["exposed_retrieval_latency_ms"] for x in r["baseline_log"]]
hoscp_lat = [x["exposed_retrieval_latency_ms"] for x in r["hoscp_log"]]
fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=200)
ax.bar([i - w/2 for i in x], base_lat, width=w, label="Baseline (synchronous retrieval)", color="#c0704d")
ax.bar([i + w/2 for i in x], hoscp_lat, width=w, label="HO-SCP (speculative prefetch)", color="#3f9a5c")
ax.set_xticks(list(x))
ax.set_xticklabels(short)
ax.set_xlabel("Manuscript section (drafting order)")
ax.set_ylabel("Generator-exposed retrieval latency (ms)")
ax.set_title("Retrieval latency exposed to the generator per section")
ax.legend()
fig.tight_layout()
fig.savefig("fig2_latency.png")
plt.close(fig)

print("figures written")
