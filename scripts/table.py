import json
import pandas as pd

with open("all_systems.json", "r") as f:
    data = json.load(f)

# Target metrics
METRICS = {
    "MRR": "mrr",
    "P@1": "precision@1",
    "P@10": "precision@10",
    "NDCG@10": "ndcg@10",
    "Latency (ms)": "retrieval_latency",
}

baseline_values = {}
system_rows = []

for system_name, entry in data.items():
    metrics = entry["metrics"]

    # Identify baseline only once
    if entry["system_a"] == "baseline" and not baseline_values:
        baseline_row = {"Version": "baseline"}
        for out_name, json_key in METRICS.items():
            metric_obj = metrics.get(json_key, {})
            baseline_row[out_name] = metric_obj.get("system_a_value")
        baseline_values = baseline_row

    # Collect system_b rows
    row = {"Version": system_name}
    for out_name, json_key in METRICS.items():
        metric_obj = metrics.get(json_key, {})
        row[out_name] = metric_obj.get("system_b_value")
    system_rows.append(row)

# Build dataframe: baseline first, then systems
rows = [baseline_values] + system_rows
df = pd.DataFrame(rows)

# Sort by MRR (baseline stays on top)
df_systems = df[df["Version"] != "baseline"].sort_values(by="MRR", ascending=False)
df = pd.concat([df[df["Version"] == "baseline"], df_systems])

# Save outputs
df.to_csv("summary_table.csv", index=False)
df.to_markdown("summary_table.md", index=False)

latex = df.to_latex(index=False, float_format="%.3f")
with open("summary_table.tex", "w") as f:
    f.write(latex)

print(df)
print("Generated summary_table.csv, summary_table.md, summary_table.tex")
