from __future__ import annotations

import random
import subprocess
from pathlib import Path
from typing import Dict

import networkx as nx


def parse_mapping_file(mapping_file: Path) -> Dict[int, int]:
    """Parse SecGraph mapping output file."""
    mapping: Dict[int, int] = {}
    if not mapping_file.exists():
        return mapping
    with mapping_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            try:
                left = int(parts[0])
                right = int(parts[1])
            except ValueError:
                continue
            mapping[left] = right
    return mapping


def run_secgraph_ns_attack(
    original_graph: nx.Graph,
    anonymized_graph: nx.Graph,
    original_pairs_path: Path,
    anonymized_pairs_path: Path,
    secgraph_jar_path: Path,
    output_dir: Path,
    run_seed: int,
    theta: float = 0.5,
) -> Dict[str, float | str]:
    """Run SecGraph NS de-anonymization attack and summarize success metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    common_nodes = sorted(set(int(n) for n in original_graph.nodes()) & set(int(n) for n in anonymized_graph.nodes()))
    if len(common_nodes) < 2:
        return {
            "deanon_success_pct": 0.0,
            "deanon_correct_mappings": 0.0,
            "deanon_eval_nodes": 0.0,
            "deanon_mapped_nonseed": 0.0,
            "attack_status": "skipped_not_enough_common_nodes",
            "attack_stdout": "",
            "attack_stderr": "",
        }

    seed_file = output_dir / "seed.txt"
    mapping_output = output_dir / "ns_mapping_output.txt"

    if mapping_output.exists() and mapping_output.stat().st_size > 0 and seed_file.exists():
        mapping = parse_mapping_file(mapping_output)
        seed_nodes = set()
        with seed_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) >= 1:
                    try:
                        seed_nodes.add(int(parts[0]))
                    except ValueError:
                        continue
        eval_nodes = set(common_nodes) - seed_nodes
        mapped_nonseed = {left: right for left, right in mapping.items() if left in eval_nodes}
        correct = sum(1 for left, right in mapped_nonseed.items() if right == left and right in eval_nodes)
        success = 100.0 * float(correct) / float(len(eval_nodes)) if eval_nodes else 0.0
        return {
            "deanon_success_pct": success,
            "deanon_correct_mappings": float(correct),
            "deanon_eval_nodes": float(len(eval_nodes)),
            "deanon_mapped_nonseed": float(len(mapped_nonseed)),
            "attack_status": "reused_existing_mapping",
            "attack_stdout": "",
            "attack_stderr": "",
        }

    rng = random.Random(run_seed)
    seed_count = max(5, int(round(0.01 * len(common_nodes))))
    seed_count = min(seed_count, max(1, len(common_nodes) - 1), 50)
    seed_nodes = set(rng.sample(common_nodes, seed_count))

    with seed_file.open("w", encoding="utf-8") as handle:
        for node in sorted(seed_nodes):
            handle.write(f"{node} {node}\n")

    command = [
        "java",
        "-Xmx8g",
        "-jar",
        str(secgraph_jar_path),
        "-m",
        "d",
        "-a",
        "NS",
        "-gA",
        str(anonymized_pairs_path),
        "-gB",
        str(original_pairs_path),
        "-seed",
        str(seed_file),
        "-theta",
        str(theta),
        "-gO",
        str(mapping_output),
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=75, check=False)
    except subprocess.TimeoutExpired:
        return {
            "deanon_success_pct": float("nan"),
            "deanon_correct_mappings": float("nan"),
            "deanon_eval_nodes": float(len(common_nodes) - len(seed_nodes)),
            "deanon_mapped_nonseed": float("nan"),
            "attack_status": "timeout",
            "attack_stdout": "",
            "attack_stderr": "Timeout",
        }

    mapping = parse_mapping_file(mapping_output)
    eval_nodes = set(common_nodes) - seed_nodes
    mapped_nonseed = {left: right for left, right in mapping.items() if left in eval_nodes}
    correct = sum(1 for left, right in mapped_nonseed.items() if right == left and right in eval_nodes)
    success = 100.0 * float(correct) / float(len(eval_nodes)) if eval_nodes else 0.0
    status = "ok" if result.returncode == 0 else f"error_code_{result.returncode}"
    return {
        "deanon_success_pct": success,
        "deanon_correct_mappings": float(correct),
        "deanon_eval_nodes": float(len(eval_nodes)),
        "deanon_mapped_nonseed": float(len(mapped_nonseed)),
        "attack_status": status,
        "attack_stdout": result.stdout.strip()[:500],
        "attack_stderr": result.stderr.strip()[:500],
    }

