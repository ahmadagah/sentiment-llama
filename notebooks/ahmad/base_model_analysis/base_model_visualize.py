import os
import json
from pathlib import Path

from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, logging as hf_logging
from safetensors.torch import load_file
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl

hf_logging.set_verbosity_error()

BASE_DIR = Path(__file__).parent
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_DIR = BASE_DIR / "Llama-3.2-1B-Instruct"

SUMMARY_TXT_PATH = BASE_DIR / f"{MODEL_DIR.name}_file_summary.txt"
SHARD_TXT_PATH = BASE_DIR / f"{MODEL_DIR.name}_shard_params.txt"
COMPONENT_PNG_PATH = BASE_DIR / f"{MODEL_DIR.name}_component_sizes.png"
SHARDS_PNG_PATH = BASE_DIR / f"{MODEL_DIR.name}_shard_params.png"

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token is None:
    raise RuntimeError("HF_TOKEN not found in .env file.")

def model_exists(path: Path) -> bool:
    return (path / "config.json").exists()

if model_exists(MODEL_DIR):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        device_map="auto",
    )
else:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=hf_token,
        device_map="auto",
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(MODEL_DIR)
    model.save_pretrained(MODEL_DIR)

def format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.2f} MB"
    else:
        return f"{size_bytes / (1024 ** 3):.2f} GB"

FILE_DESCRIPTIONS = {
    "tokenizer.json": "Core tokenizer vocabulary and merge rules.",
    "tokenizer_config.json": "Tokenizer configuration.",
    "special_tokens_map.json": "Mapping of special tokens.",
    "config.json": "Model architecture definition.",
    "generation_config.json": "Default text generation settings.",
    "model.safetensors.index.json": "Index for safetensors shard files.",
}

def get_file_description(name: str) -> str:
    if name in FILE_DESCRIPTIONS:
        return FILE_DESCRIPTIONS[name]
    if name.startswith("model-") and name.endswith(".safetensors"):
        return "Shard of model weights (safetensors)."
    return "Other file in the model directory."

def get_shard_param_counts(model_dir: Path) -> dict:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        return {}
    with open(index_path, "r") as f:
        index_data = json.load(f)
    weight_map = index_data.get("weight_map", {})
    shard_to_tensors = {}
    for tensor_name, shard_name in weight_map.items():
        shard_to_tensors.setdefault(shard_name, []).append(tensor_name)
    shard_param_counts = {}
    for shard_name, tensor_list in shard_to_tensors.items():
        shard_path = model_dir / shard_name
        tensors = load_file(shard_path)
        total_params = 0
        for name in tensor_list:
            t = tensors[name]
            total_params += t.numel()
        shard_param_counts[shard_name] = total_params
    return shard_param_counts

shard_param_counts = get_shard_param_counts(MODEL_DIR)

def build_model_dir_table(model_dir: Path, shard_param_counts: dict):
    rows = []
    for path in sorted(model_dir.glob("*")):
        if not path.is_file():
            continue
        if path.name == "chat_template.jinja":
            continue
        size_bytes = path.stat().st_size
        size_str = format_size(size_bytes)
        params = shard_param_counts.get(path.name)
        if params is None:
            params_str = "-"
        else:
            params_str = f"{params:,} (~{params / 1e9:.3f} B)"
        desc = get_file_description(path.name)
        rows.append((path.name, size_str, params_str, desc))
    return rows

def save_model_dir_summary(rows, txt_path: Path, model_stats_lines):
    header = f"{'Filename':35s} {'Size':>12s}  {'Parameters':>24s}  Description"
    line = "-" * len(header)
    lines_out = ["Llama 3.2 1B Instruct file summary", "", header, line]
    for name, size_str, params_str, desc in rows:
        row_str = f"{name:35s} {size_str:>12s}  {params_str:>24s}  {desc}"
        lines_out.append(row_str)
    lines_out.append("")
    lines_out.append("Model-level stats")
    lines_out.append("-----------------")
    lines_out.extend(model_stats_lines)
    txt_path.write_text("\n".join(lines_out), encoding="utf-8")

total_params = sum(p.numel() for p in model.parameters())
total_params_b = total_params / 1e9
config = model.config

vocab_size = getattr(config, "vocab_size", None)
hidden_size = getattr(config, "hidden_size", None)
num_layers = getattr(config, "num_hidden_layers", None)
num_heads = getattr(config, "num_attention_heads", None)
max_seq_len = getattr(config, "max_position_embeddings", None)

model_stats_lines = [
    f"MODEL_ID: {MODEL_ID}",
    f"Total parameters: {total_params:,} (~{total_params_b:.3f} B)",
    f"Vocab size: {vocab_size}",
    f"Hidden size: {hidden_size}",
    f"# Layers: {num_layers}",
    f"# Attention heads: {num_heads}",
    f"Max sequence length: {max_seq_len}",
]

rows = build_model_dir_table(MODEL_DIR, shard_param_counts)
save_model_dir_summary(rows, SUMMARY_TXT_PATH, model_stats_lines)

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
})

def collect_file_stats(model_dir: Path, shard_param_counts: dict):
    names = []
    sizes_gb = []
    pretty_sizes = []
    params_b = []
    for path in sorted(model_dir.glob("*")):
        if not path.is_file():
            continue
        if path.name == "chat_template.jinja":
            continue
        size_bytes = path.stat().st_size
        size_gb = size_bytes / (1024 ** 3)
        names.append(path.name)
        sizes_gb.append(size_gb)
        pretty_sizes.append(format_size(size_bytes))
        p = shard_param_counts.get(path.name)
        params_b.append(p / 1e9 if p is not None else None)
    return names, sizes_gb, pretty_sizes, params_b

def plot_model_component_sizes(names, sizes_gb, pretty_sizes, png_path: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    y_pos = range(len(names))
    cmap = plt.cm.get_cmap("tab10")
    colors = [cmap(i % cmap.N) for i in y_pos]
    ax.barh(y_pos, sizes_gb, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Size (GB)")
    ax.set_title(f"Storage footprint of {MODEL_DIR.name} components")
    for i, (width, label) in enumerate(zip(sizes_gb, pretty_sizes)):
        ax.text(width + 0.02, i, label, va="center", fontsize=9)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

def plot_shard_parameters(shard_param_counts: dict, png_path: Path, txt_path: Path):
    if not shard_param_counts:
        return
    shard_names = list(shard_param_counts.keys())
    params = [shard_param_counts[name] for name in shard_names]
    params_b = [p / 1e9 for p in params]
    total = sum(params)
    total_b = total / 1e9
    fig, ax = plt.subplots(figsize=(7, 3.6))
    y_pos = range(len(shard_names))
    cmap = plt.cm.get_cmap("tab20")
    colors = [cmap(i * 3 % cmap.N) for i in y_pos]
    bar_height = 0.28
    ax.barh(
        y_pos,
        params_b,
        color=colors,
        height=bar_height,
        edgecolor="black",
        linewidth=0.6,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(shard_names)
    ax.invert_yaxis()
    ax.set_xlabel("Parameters (billions)")
    ax.set_title(
        f"Parameter distribution across checkpoint shards\n"
        f"({MODEL_DIR.name}, total ≈ {total_b:.3f} B)"
    )
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.6)
    for i, (v_b, v_raw) in enumerate(zip(params_b, params)):
        pct = 100.0 * v_raw / total
        ax.text(
            v_b + 0.02,
            i,
            f"{v_b:.3f} B ({pct:.1f}%)",
            va="center",
            ha="left",
            fontsize=9,
        )
    plt.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    lines = [
        f"Shard parameter breakdown (total {total:,} ≈ {total_b:.3f} B params):"
    ]
    for name, v_raw, v_b in zip(shard_names, params, params_b):
        pct = 100.0 * v_raw / total
        lines.append(f"{name}: {v_raw:,} params ≈ {v_b:.3f} B ({pct:.1f}%)")
    txt_path.write_text("\n".join(lines), encoding="utf-8")

names, sizes_gb, pretty_sizes, params_b = collect_file_stats(MODEL_DIR, shard_param_counts)
plot_model_component_sizes(names, sizes_gb, pretty_sizes, COMPONENT_PNG_PATH)
plot_shard_parameters(shard_param_counts, SHARDS_PNG_PATH, SHARD_TXT_PATH)

print("Done.")
