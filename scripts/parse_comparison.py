import json
import re
from pathlib import Path


# ----------------------------
# SINGLE FILE PARSER
# ----------------------------
def parse_comparison_file(path):
    text = Path(path).read_text()

    # 1. Extract systems
    header_match = re.search(
        r"Metric\s+(\S+)\s+(\S+)\s+Best",
        text
    )
    if not header_match:
        raise ValueError(f"Could not read system names in: {path}")

    system_a = header_match.group(1)
    system_b = header_match.group(2)

    # 2. Metric comparison block
    metric_section = re.search(
        r"METRIC COMPARISON:(.*?)IMPROVEMENT ANALYSIS",
        text,
        flags=re.S
    )
    if not metric_section:
        raise ValueError(f"Could not find METRIC COMPARISON in: {path}")

    metrics_text = metric_section.group(1)
    metrics = {}

    line_pattern = re.compile(
        r"([\w@]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\S+)"
    )

    for metric, a, b, best in line_pattern.findall(metrics_text):
        metrics[metric] = {
            "system_a_value": float(a),
            "system_b_value": float(b),
            "best": best
        }

    # 3. Improvement block
    improv_section = re.search(
        r"IMPROVEMENT ANALYSIS.*?:\s*(.*)",
        text,
        flags=re.S
    )

    improvements = {}
    if improv_section:
        improv_text = improv_section.group(1)

        block_pattern = re.compile(
            r"([\w@]+):\s*baseline:\s*([0-9.\-]+)\s*"
            r".*?:\s*([0-9.\-]+)\s*"
            r"Improvement:\s*([0-9.\-]+)\s*\(([0-9.\-]+)%\)",
            flags=re.S
        )

        for metric, a, b, delta, pct in block_pattern.findall(improv_text):
            improvements[metric] = {
                "baseline": float(a),
                "other": float(b),
                "delta": float(delta),
                "percent": float(pct)
            }

    return {
        "system_a": system_a,
        "system_b": system_b,
        "metrics": metrics,
        "improvements": improvements
    }


# ----------------------------
# MAIN BATCH SCRIPT
# ----------------------------
def batch_parse(root="data/chunk_sweep"):
    root_path = Path(root)
    result = {}

    for comp_file in root_path.rglob("comparison.txt"):
        system_name = comp_file.parent.name  # folder name
        print(f"Parsing: {system_name}")

        try:
            parsed = parse_comparison_file(comp_file)
            result[system_name] = parsed
        except Exception as e:
            print(f"❌ Failed to parse {comp_file}: {e}")

    # Save giant JSON
    output_path = Path("all_systems.json")
    output_path.write_text(json.dumps(result, indent=2))

    print(f"\n✅ Done! Saved: {output_path.resolve()}")


if __name__ == "__main__":
    batch_parse()
