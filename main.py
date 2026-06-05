import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx

import torch
from torch_geometric.utils import from_networkx

from transformers import AutoModelForCausalLM, AutoTokenizer

from tuners.cgp_tuning.config import GraphPromptEncoderConfig
from tuners.peft_model import GraphPeftModelForCausalLM

from utils.data_utils import read_file, DataCollatorForChatML
from utils.json_utils import read_json_file, write_json_file


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_metadata_log(metadata_path: str) -> Dict[int, dict]:
    """Load the conversion metadata JSONL file, keyed by sample idx."""
    meta_by_idx: Dict[int, dict] = {}
    if not os.path.exists(metadata_path):
        return meta_by_idx
    with open(metadata_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            idx = entry.get("idx")
            if idx is not None:
                meta_by_idx[idx] = entry
    return meta_by_idx


def process_one_sample(
    sample_path: str,
    peft_model: GraphPeftModelForCausalLM,
    tokenizer,
    message_template: dict,
    node_type_to_index: dict,
    edge_type_to_index: dict,
    device: str,
    collator: DataCollatorForChatML,
) -> Dict[str, Any]:
    """Run inference on a single converted sample (.json).

    Returns a dict with ``idx``, ``true_logit``, ``false_logit``,
    ``pred_target``, ``true_target``, and any error info.
    """
    data_sample = read_json_file(sample_path)

    # -- graph -----------------------------------------------------------
    graph_data = data_sample["graph"]
    if "links" in graph_data and "edges" not in graph_data:
        graph_data["edges"] = graph_data.pop("links")
    graph = from_networkx(nx.json_graph.node_link_graph(graph_data))
    graph.label = torch.tensor(
        [node_type_to_index.get(label, node_type_to_index["UNKNOWN"])
         for label in graph.label]
    )
    graph.edge_label = torch.tensor(
        [edge_type_to_index.get(edge_label, -1)
         for edge_label in graph.edge_label]
    )
    # Keep only edges whose type is in the index
    valid_edge_mask = graph.edge_label >= 0
    if not valid_edge_mask.all():
        graph.edge_index = graph.edge_index[:, valid_edge_mask]
        graph.edge_label = graph.edge_label[valid_edge_mask]
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.int64)

    # -- text ------------------------------------------------------------
    messages_key = "messages"
    examples = [{
        messages_key: [
            {
                "role": "user",
                "content": message_template["user"].format(
                    func=data_sample["func"]
                ),
            },
            {
                "role": "assistant",
                "content": message_template["assistant"].format(
                    target=data_sample["target"]
                ),
            },
        ]
    }]
    data = collator(examples)

    true_target = data_sample["target"]

    # -- forward ---------------------------------------------------------
    with torch.no_grad():
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        with torch.autocast(device_type=device, dtype=dtype):
            outputs = peft_model(
                input_ids=data["prompts"].to(device),
                attention_mask=data["prompt_attention_mask"].to(device),
                graphs=graph.to(device),
            )
        logits = outputs.logits[:, -1, :]
        true_id = tokenizer.convert_tokens_to_ids("true")
        false_id = tokenizer.convert_tokens_to_ids("false")
        true_logit = logits[0, true_id].item()
        false_logit = logits[0, false_id].item()
        pred = true_logit > false_logit

    return {
        "true_target": true_target,
        "pred_target": bool(pred),
        "correct": bool(pred) == bool(true_target),
        "true_logit": true_logit,
        "false_logit": false_logit,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CGP-Tuning inference for vulnerability detection."
    )
    parser.add_argument(
        "--data-dir",
        help="Directory of converted .json samples (batch mode).",
    )
    parser.add_argument(
        "--single-file",
        help="Path to a single converted .json sample.",
    )
    parser.add_argument(
        "--metadata-log",
        default="processed_data/conversion_metadata.jsonl",
        help="Path to the conversion metadata JSONL (for rich logging).",
    )
    parser.add_argument(
        "--output-log",
        default="inference_results.jsonl",
        help="Where to write per-sample inference results (JSONL).",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Max samples to evaluate (0 = all).",
    )
    parser.add_argument(
        "--base-model-dir", default="/root/workspace/Qwen2.5-Coder-7B",
        help="Path to the base LLM.",
    )
    parser.add_argument(
        "--templates-dir", default="templates",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    torch.autograd.set_detect_anomaly(True)
    torch.backends.mkl.enabled = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("number of gpus", torch.cuda.device_count())
    print("number of cpus", os.cpu_count())
    print(f"using device: {device}")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stderr),
        ],
    )
    logger = logging.getLogger("inference")

    # ------------------------------------------------------------------
    # Load model once
    # ------------------------------------------------------------------
    base_model_dir = args.base_model_dir
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
    tokenizer.chat_template = read_file(
        os.path.join(args.templates_dir, "chat_template.jinja")
    )
    tokenizer.pad_token = (
        tokenizer.bos_token if tokenizer.pad_token is None else tokenizer.pad_token
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_dir, torch_dtype="auto", trust_remote_code=True
    )
    base_model.gradient_checkpointing_enable()
    for name, module in base_model.named_children():
        for param in module.parameters():
            param.requires_grad = False

    config = GraphPromptEncoderConfig(
        peft_type="CGP_TUNING",
        task_type="CAUSAL_LM",
        inference_mode=False,
        base_model_name_or_path=base_model_dir,
        token_dim=base_model.config.hidden_size,
        gnn_input_size=base_model.config.hidden_size,
        gnn_hidden_size=base_model.config.hidden_size,
        gnn_output_size=base_model.config.hidden_size,
        gnn_num_layers=4,
        gnn_num_heads=8,
        gnn_dropout=0.1,
        gnn_bias=True,
        num_node_types=18,
        num_edge_types=13,
        max_num_nodes=4096,
        edge_dim=128,
        ablate_node_type_embeddings=False,
        ablate_edge_type_embeddings=False,
        ablate_positional_embeddings=False,
        num_virtual_tokens=32,
        cma_hidden_size=base_model.config.hidden_size,
        cma_num_heads=32,
        cma_num_key_value_heads=32,
        cma_dropout=0.1,
        cma_bias=False,
        ablate_cross_modal_alignment_module=False,
        ablate_multi_head_attn=False,
    )
    peft_model = GraphPeftModelForCausalLM(config=config, base_model=base_model)
    peft_model.print_trainable_parameters()
    peft_model.eval()
    peft_model.to(device)

    # ------------------------------------------------------------------
    # Shared resources
    # ------------------------------------------------------------------
    node_type_to_index = read_json_file(
        os.path.join(args.templates_dir, "node_type_to_index.json")
    )
    edge_type_to_index = read_json_file(
        os.path.join(args.templates_dir, "edge_type_to_index.json")
    )
    message_template = read_json_file(
        os.path.join(args.templates_dir, "message_template.json")
    )
    collator = DataCollatorForChatML(
        tokenizer=tokenizer, max_length=16_000, ignore_index=-100,
        messages_key="messages", is_padding=True,
    )

    # Load metadata from conversion for rich logging
    metadata_by_idx = load_metadata_log(args.metadata_log)
    if metadata_by_idx:
        logger.info("Loaded metadata for %d samples from %s",
                     len(metadata_by_idx), args.metadata_log)

    # ------------------------------------------------------------------
    # Collect samples
    # ------------------------------------------------------------------
    sample_paths: List[str] = []

    if args.single_file:
        sample_paths = [args.single_file]
    elif args.data_dir:
        data_dir = Path(args.data_dir)
        sample_paths = sorted(str(p) for p in data_dir.glob("*.json")
                              if not p.name.startswith("conversion_"))
    else:
        # Backward-compatible: single example.json in cwd
        fallback = "example.json"
        if os.path.exists(fallback):
            sample_paths = [fallback]
        else:
            logger.error(
                "No input specified. Use --data-dir, --single-file, or place "
                "an example.json in the working directory."
            )
            sys.exit(1)

    if args.limit > 0:
        sample_paths = sample_paths[: args.limit]

    total = len(sample_paths)
    logger.info("Evaluating %d sample(s)", total)

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------
    results: List[dict] = []
    correct = 0

    for i, sp in enumerate(sample_paths, 1):
        fname = Path(sp).name
        # Parse idx from filename like "194963_vul.json"
        sample_idx = None
        stem = Path(sp).stem  # e.g. "194963_vul"
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1] in ("vul", "safe"):
            sample_idx = int(parts[0])

        logger.info("[%d/%d] %s", i, total, fname)

        try:
            pred = process_one_sample(
                sp, peft_model, tokenizer, message_template,
                node_type_to_index, edge_type_to_index, device, collator,
            )
        except Exception:
            logger.exception("FAILED: %s", fname)
            pred = {"error": str(sys.exc_info()[1])}

        # --- Build rich result entry -----------------------------------
        entry: Dict[str, Any] = {
            "file": fname,
            "sample_idx": sample_idx,
        }

        # Merge original dataset metadata if available
        if sample_idx is not None and sample_idx in metadata_by_idx:
            meta = metadata_by_idx[sample_idx]
            entry["project"] = meta.get("project")
            entry["file_name"] = meta.get("file_name")
            entry["cwe"] = meta.get("cwe")
            entry["cve"] = meta.get("cve")
            entry["cve_desc"] = meta.get("cve_desc")
            entry["nvd_url"] = meta.get("nvd_url")
            entry["commit_id"] = meta.get("commit_id")
            entry["language"] = meta.get("language")
            entry["vul_file_hash"] = (
                meta.get("vul", {}).get("file_hash")
            )
            entry["safe_file_hash"] = (
                meta.get("safe", {}).get("file_hash")
            )

        entry.update(pred)

        if pred.get("correct"):
            correct += 1

        results.append(entry)
        logger.info(
            "  true=%s  pred=%s  correct=%s  true_logit=%.4f  false_logit=%.4f",
            pred.get("true_target"),
            pred.get("pred_target"),
            pred.get("correct"),
            pred.get("true_logit", float("nan")),
            pred.get("false_logit", float("nan")),
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    accuracy = correct / total if total > 0 else 0.0
    logger.info("=" * 60)
    logger.info("Accuracy: %d/%d = %.4f", correct, total, accuracy)

    # Write detailed results
    out_log = args.output_log
    with open(out_log, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    logger.info("Detailed results written to %s", out_log)

    # Also write a compact summary
    summary_path = out_log.replace(".jsonl", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "base_model_dir": base_model_dir,
            "device": device,
        }, f, indent=2)
    logger.info("Summary written to %s", summary_path)
