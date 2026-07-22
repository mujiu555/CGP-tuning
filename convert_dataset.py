#!/usr/bin/env python3
"""
convert_dataset.py — Convert the RA-ILLK GraphSON dataset to the node-link
graph format expected by the CGP-tuning project.

Dataset format (input):
  Each .json file under --dataset-dir contains:
    - idx, project, file_name, cwe, cve, language
    - commit_id, commit_url, cve_desc, nvd_url
    - issue_code:  { source, cpg_graphson, file_hash }   ← vulnerable version
    - fixed_code:  { source, cpg_graphson, file_hash }   ← safe/patched version

Target format (output):
  Two .json files per dataset entry: {idx}_vul.json and {idx}_safe.json
  Each contains:
    - func:   source code string
    - target: 1 (vulnerable) or 0 (safe)
    - graph:  networkx node-link graph { directed, multigraph, graph, nodes, links }

  Additionally, a metadata log (JSONL) is written with the full original
  dataset fields plus processing statistics so nothing is lost.

Usage:
  python convert_dataset.py --dataset-dir /path/to/test_graphson
                            [--output-dir ./processed]
                            [--use-joern]           # re-generate CPG via Joern
                            [--joern-bin joern-parse]
                            [--log-file metadata.jsonl]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Node / edge type index mirrors (must stay in sync with
# templates/node_type_to_index.json and templates/edge_type_to_index.json)
# ---------------------------------------------------------------------------

def _load_index(path: str) -> Dict[str, int]:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# GraphSON → node-link conversion
# ---------------------------------------------------------------------------

def _extract_prop_value(prop: dict) -> str:
    """Extract a scalar string value from a GraphSON g:VertexProperty dict.

    GraphSON wraps scalar properties as::

        {
          "@type": "g:VertexProperty",
          "@value": {
            "@type": "g:List",
            "@value": [ "<actual value>" ]
          }
        }

    Returns the first list element as a string (or ``""`` on failure).
    """
    try:
        inner = prop["@value"]                 # { "@type": "g:List", "@value": [...] }
        items = inner["@value"]                # [ elem, ... ]
        if not items:
            return ""
        val = items[0]
        if isinstance(val, dict):
            # e.g. {"@type": "g:Int32", "@value": 42}
            return str(val.get("@value", ""))
        return str(val)
    except (KeyError, IndexError, TypeError):
        return ""


def graphson_to_node_link(
    graphson_str: str,
    node_type_index: Dict[str, int],
    edge_type_index: Dict[str, int],
    logger: logging.Logger,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Convert a single Joern GraphSON string into the project's node-link dict.

    Returns
    -------
    (graph_dict, skipped_edge_counts)
        graph_dict is ready to be serialised as the ``"graph"`` field.
        skipped_edge_counts tallies edge labels that had no mapping.
    """
    cpg = json.loads(graphson_str)

    vertices: List[dict] = cpg["@value"]["vertices"]
    edges: List[dict] = cpg["@value"]["edges"]

    # -- vertices -----------------------------------------------------------
    nodes: List[dict] = []
    unmapped_vlabels: Counter[str] = Counter()

    for v in vertices:
        vid = str(v["id"]["@value"])
        vlabel = v["label"]

        if vlabel not in node_type_index:
            unmapped_vlabels[vlabel] += 1
            mapped_label = "UNKNOWN"
        else:
            mapped_label = vlabel

        # CODE property  (most nodes have it; TYPE / BINDING / META_DATA don't)
        props = v.get("properties", {})
        code_val = _extract_prop_value(props["CODE"]) if "CODE" in props else ""

        nodes.append({
            "label": mapped_label,
            "CODE": code_val,
            "id": vid,
        })

    if unmapped_vlabels:
        logger.warning(
            "Unmapped vertex labels (→ UNKNOWN): %s",
            dict(unmapped_vlabels),
        )

    # -- edges --------------------------------------------------------------
    links: List[dict] = []
    skipped_elabels: Counter[str] = Counter()

    for e in edges:
        elabel = e["label"]
        if elabel not in edge_type_index:
            skipped_elabels[elabel] += 1
            continue

        links.append({
            "label": elabel,
            "source": str(e["outV"]["@value"]),
            "target": str(e["inV"]["@value"]),
            "key": e["id"]["@value"],
        })

    if skipped_elabels:
        logger.info("Skipped edge labels (not in project index): %s", dict(skipped_elabels))

    graph = {
        "directed": True,
        "multigraph": True,
        "graph": {"name": "G"},
        "nodes": nodes,
        "links": links,
    }
    return graph, dict(skipped_elabels)


# ---------------------------------------------------------------------------
# Joern interface (optional — invoked when --use-joern is passed)
# ---------------------------------------------------------------------------

def _find_joern_bin(requested: Optional[str]) -> str:
    """Resolve the joern-parse binary path."""
    if requested:
        if os.path.isfile(requested):
            return requested
        raise FileNotFoundError(f"Joern binary not found: {requested}")
    # Try common names / locations
    for candidate in ("joern-parse", "joern"):
        p = subprocess.run(
            ["which", candidate], capture_output=True, text=True
        )
        if p.returncode == 0:
            return p.stdout.strip()
    raise RuntimeError(
        "Cannot locate joern-parse. Install Joern or pass --joern-bin."
    )


def run_joern_parse(source_code: str, language: str, joern_bin: str) -> str:
    """Run Joern on *source_code* and return the GraphSON JSON string.

    1. Write source to a temp ``.c`` file.
    2. Run ``joern-parse --language <lang> -o <cpg.bin> <file>``
    3. Run ``joern-export --repr graphson --out <outdir> <cpg.bin>``
    4. Read and return the exported GraphSON.
    """
    lang_flag = {"c": "c", "c++": "c", "cpp": "c"}.get(
        language.lower(), language.lower()
    )

    with tempfile.TemporaryDirectory(prefix="joern_") as tmp:
        src_path = os.path.join(tmp, f"code.{'cpp' if lang_flag == 'c' else lang_flag}")
        with open(src_path, "w") as f:
            f.write(source_code)

        cpg_bin = os.path.join(tmp, "cpg.bin")
        out_dir = os.path.join(tmp, "export")

        # 1. Parse
        parse_cmd = [
            joern_bin,
            f"--language={lang_flag}",
            "-o", cpg_bin,
            src_path,
        ]
        subprocess.run(parse_cmd, check=True, capture_output=True, text=True)

        # 2. Export
        export_cmd = [
            "joern-export",
            "--repr", "graphson",
            "--out", out_dir,
            cpg_bin,
        ]
        subprocess.run(export_cmd, check=True, capture_output=True, text=True)

        # 3. Read exported GraphSON  (joern-export writes one .json per CPG)
        exported = list(Path(out_dir).glob("*.json"))
        if not exported:
            raise RuntimeError("Joern export produced no output files")
        return exported[0].read_text()


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_dataset_entry(
    entry_path: str,
    node_type_index: Dict[str, int],
    edge_type_index: Dict[str, int],
    output_dir: str,
    logger: logging.Logger,
    use_joern: bool = False,
    joern_bin: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert one dataset JSON file into two output samples (vul + safe).

    Returns a metadata dict for logging.
    """
    with open(entry_path) as f:
        raw = json.load(f)

    idx = raw["idx"]
    metadata = {
        "idx": idx,
        "project": raw.get("project"),
        "file_name": raw.get("file_name"),
        "cwe": raw.get("cwe"),
        "cve": raw.get("cve"),
        "language": raw.get("language"),
        "commit_id": raw.get("commit_id"),
        "commit_url": raw.get("commit_url"),
        "cve_desc": raw.get("cve_desc"),
        "nvd_url": raw.get("nvd_url"),
    }

    # --- issue_code (vulnerable, target = 1) -------------------------------
    issue = raw["issue_code"]
    issue_cpg_str = (issue.get("cpg_graphson") or "").strip()
    if use_joern and joern_bin:
        logger.info("[%s] vul  – regenerating CPG via Joern", idx)
        issue_cpg_str = run_joern_parse(
            issue["source"], metadata["language"], joern_bin
        )

    if not issue_cpg_str:
        logger.warning("[%s] vul  – SKIPPED (empty cpg_graphson)", idx)
        metadata["vul"] = {
            "file": None,
            "num_nodes": 0,
            "num_edges": 0,
            "skipped_edge_labels": {},
            "file_hash": issue.get("file_hash"),
            "skipped": True,
            "reason": "empty cpg_graphson",
        }
    else:
        vul_graph, vul_skipped = graphson_to_node_link(
            issue_cpg_str, node_type_index, edge_type_index, logger
        )
        vul_out = {
            "func": issue["source"],
            "target": 1,
            "graph": vul_graph,
        }
        vul_path = os.path.join(output_dir, f"{idx}_vul.json")
        with open(vul_path, "w") as f:
            json.dump(vul_out, f)
        logger.info(
            "[%s] vul  → %s  (nodes=%d, edges=%d)",
            idx,
            vul_path,
            len(vul_graph["nodes"]),
            len(vul_graph["links"]),
        )
        metadata["vul"] = {
            "file": f"{idx}_vul.json",
            "num_nodes": len(vul_graph["nodes"]),
            "num_edges": len(vul_graph["links"]),
            "skipped_edge_labels": vul_skipped,
            "file_hash": issue.get("file_hash"),
        }

    # --- fixed_code (safe, target = 0) ------------------------------------
    fixed = raw["fixed_code"]
    fixed_cpg_str = (fixed.get("cpg_graphson") or "").strip()
    if use_joern and joern_bin:
        logger.info("[%s] safe – regenerating CPG via Joern", idx)
        fixed_cpg_str = run_joern_parse(
            fixed["source"], metadata["language"], joern_bin
        )

    if not fixed_cpg_str:
        logger.warning("[%s] safe – SKIPPED (empty cpg_graphson)", idx)
        metadata["safe"] = {
            "file": None,
            "num_nodes": 0,
            "num_edges": 0,
            "skipped_edge_labels": {},
            "file_hash": fixed.get("file_hash"),
            "skipped": True,
            "reason": "empty cpg_graphson",
        }
    else:
        safe_graph, safe_skipped = graphson_to_node_link(
            fixed_cpg_str, node_type_index, edge_type_index, logger
        )
        safe_out = {
            "func": fixed["source"],
            "target": 0,
            "graph": safe_graph,
        }
        safe_path = os.path.join(output_dir, f"{idx}_safe.json")
        with open(safe_path, "w") as f:
            json.dump(safe_out, f)
        logger.info(
            "[%s] safe → %s  (nodes=%d, edges=%d)",
            idx,
            safe_path,
            len(safe_graph["nodes"]),
            len(safe_graph["links"]),
        )
        metadata["safe"] = {
            "file": f"{idx}_safe.json",
            "num_nodes": len(safe_graph["nodes"]),
            "num_edges": len(safe_graph["links"]),
            "skipped_edge_labels": safe_skipped,
            "file_hash": fixed.get("file_hash"),
        }

    return metadata


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert RA-ILLK GraphSON dataset to CGP-tuning format."
    )
    parser.add_argument(
        "--dataset-dir", required=True,
        help="Path to the dataset directory containing .json files.",
    )
    parser.add_argument(
        "--output-dir", default="./processed_data",
        help="Directory to write converted samples (default: ./processed_data).",
    )
    parser.add_argument(
        "--templates-dir", default="./templates",
        help="Path to the templates/ directory with node/edge type indexes.",
    )
    parser.add_argument(
        "--use-joern", action="store_true",
        help="Re-generate CPG via Joern instead of using the embedded cpg_graphson.",
    )
    parser.add_argument(
        "--joern-bin",
        help="Path to joern-parse binary (auto-detected if omitted).",
    )
    parser.add_argument(
        "--log-file", default="conversion_metadata.jsonl",
        help="Where to write the metadata JSONL log (default: conversion_metadata.jsonl).",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Process only the first N files (0 = all).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(
                os.path.join(args.output_dir, "conversion.log"),
                mode="w",
            ),
        ],
    )
    logger = logging.getLogger("convert_dataset")

    # Load type indexes
    node_type_index = _load_index(
        os.path.join(args.templates_dir, "node_type_to_index.json")
    )
    edge_type_index = _load_index(
        os.path.join(args.templates_dir, "edge_type_to_index.json")
    )

    logger.info("Node types: %d  Edge types: %d",
                len(node_type_index), len(edge_type_index))

    # Joern (optional)
    joern_bin = None
    if args.use_joern:
        joern_bin = _find_joern_bin(args.joern_bin)
        logger.info("Using Joern binary: %s", joern_bin)

    # ------------------------------------------------------------------
    # Collect input files
    # ------------------------------------------------------------------
    dataset_dir = Path(args.dataset_dir)
    json_files = sorted(
        p for p in dataset_dir.glob("*.json") if p.name != args.log_file
    )
    if args.limit > 0:
        json_files = json_files[: args.limit]

    logger.info("Found %d dataset files in %s", len(json_files), dataset_dir)
    if not json_files:
        logger.error("No .json files found in %s", dataset_dir)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Process
    # ------------------------------------------------------------------
    all_metadata: List[dict] = []
    success = 0
    failed = 0

    for i, filepath in enumerate(json_files, 1):
        logger.info("[%d/%d] Processing %s ...", i, len(json_files), filepath.name)
        try:
            meta = process_dataset_entry(
                str(filepath),
                node_type_index,
                edge_type_index,
                args.output_dir,
                logger,
                use_joern=args.use_joern,
                joern_bin=joern_bin,
            )
            all_metadata.append(meta)
            success += 1
        except Exception:
            logger.exception("FAILED: %s", filepath.name)
            failed += 1
            all_metadata.append({
                "file": filepath.name,
                "status": "FAILED",
                "error": str(sys.exc_info()[1]),
            })

    # ------------------------------------------------------------------
    # Write metadata log (JSONL — one JSON object per line)
    # ------------------------------------------------------------------
    log_path = os.path.join(args.output_dir, args.log_file)
    with open(log_path, "w") as f:
        for m in all_metadata:
            f.write(json.dumps(m) + "\n")

    # Also write a human-readable summary
    summary_path = os.path.join(args.output_dir, "conversion_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "dataset_dir": str(dataset_dir),
            "output_dir": args.output_dir,
            "total_files": len(json_files),
            "success": success,
            "failed": failed,
            "use_joern": args.use_joern,
        }, f, indent=2)

    logger.info("=" * 60)
    logger.info("DONE  success=%d  failed=%d  total=%d", success, failed, len(json_files))
    logger.info("Metadata log : %s", log_path)
    logger.info("Summary      : %s", summary_path)
    logger.info("Output dir   : %s", args.output_dir)


if __name__ == "__main__":
    main()
