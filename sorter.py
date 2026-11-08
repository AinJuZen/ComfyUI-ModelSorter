#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""comfyui-model-sorter —— 按文件内容判断这是哪种模型，并放进正确的目录。

不靠文件名猜。safetensors 的头里写着每个张量的名字，
那些名字就是这套结构的"指纹"：有 lora_down/lora_up 的是 LoRA，
有 encoder.down + decoder.up 的是 VAE，有 double_blocks 的是 Flux 主干……

用法：
    python sorter.py detect  <文件或目录> ...
    python sorter.py place   <文件或目录> ... --models-root <ComfyUI/models>
    python sorter.py rules

    --dry-run    只看会放哪，不动文件
    --json       机器可读输出
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
from pathlib import Path

# ---------------------------------------------------------------- 规则表

MODEL_SUFFIXES = {".safetensors", ".sft", ".ckpt", ".pt", ".pt2", ".bin", ".pth", ".gguf", ".pkl"}
TEMP_SUFFIXES = {".xltd", ".td", ".part", ".crdownload", ".downloading", ".tmp"}

FOLDER_LABELS = {
    "checkpoints": "大模型 / 一体化 checkpoint",
    "diffusion_models": "扩散主干（UNet / DiT）",
    "text_encoders": "文本编码器",
    "vae": "VAE",
    "loras": "LoRA",
    "controlnet": "ControlNet",
    "clip_vision": "CLIP Vision",
    "upscale_models": "放大模型",
    "embeddings": "Embedding",
    "audio_encoders": "音频编码器",
}

# 公开的规则表（文档和 --rules 共用一份）
RULES = [
    ("lora_unet / lora_te / lora_down. / lora_up. / lora_a. / lora_b.", "loras"),
    ("model.diffusion_model + (vae. / first_stage_model / cond_stage_model / text_encoders.)", "checkpoints"),
    ("model.diffusion_model（单独出现）", "diffusion_models"),
    ("double_blocks / single_blocks", "diffusion_models（Flux 主干）"),
    ("cap_embedder / noise_refiner / context_refiner", "diffusion_models（DiT）"),
    ("layers. + (proj.qkv / text_embedder. / x_embedder / t_embedder / patch_embed / adaln)", "diffusion_models（DiT，如 MOSS-Image）"),
    ("blocks. + (attn.wq / attn.wk / attn.wv / mlp.gate)", "diffusion_models（DiT，如 Krea / Chroma）"),
    ("encoder.down* 且 decoder.up*", "vae"),
    ("model.language_model. / model.embed_tokens.", "text_encoders（LLM / VLM 结构）"),
    ("model.layers + model.embed_tokens", "text_encoders（Qwen 类）"),
    ("text_model. / text_encoder. / cond_stage_model.transformer / encoder.block.", "text_encoders（CLIP / T5）"),
    ("control_model / input_hint_block / controlnet_cond_embedding / zero_convs", "controlnet"),
    ("vision_model. / visual.", "clip_vision"),
    ("emb_params / string_to_token / string_to_param", "embeddings"),
]


# ---------------------------------------------------------------- 读头

def read_header_keys(path: Path):
    """只读 safetensors 的 JSON 头部，返回张量名列表；读不了返回 None。

    不加载权重，只读前 8 字节的长度 + 紧跟着的 JSON，几百 KB 的 IO。
    """
    try:
        with open(path, "rb") as f:
            head = f.read(8)
            if len(head) < 8:
                return None
            size = struct.unpack("<Q", head)[0]
            if size <= 0 or size > 200_000_000:
                return None
            raw = f.read(size)
            obj = json.loads(raw.decode("utf-8"))
        return [k for k in obj.keys() if k != "__metadata__"]
    except Exception:
        return None


# ---------------------------------------------------------------- 判定

def detect_from_keys(keys):
    """按张量名判类型。返回 (目录, 依据) 或 (None, None)。"""
    low = [k.lower() for k in keys]

    def has(*needles):
        return any(n in k for k in low for n in needles)

    def starts(*prefixes):
        return any(k.startswith(p) for k in low for p in prefixes)

    if has("lora_unet", "lora_te", "lora_down.", "lora_up.", "lora_a", "lora_b"):
        return "loras", "含 LoRA 权重（lora_down / lora_up 或 lora_unet / lora_te）"

    if starts("model.diffusion_model"):
        if starts("vae.", "first_stage_model", "cond_stage_model", "text_encoders."):
            return "checkpoints", "同时含扩散主干 + VAE/文本编码器 → 一体化大模型"
        return "diffusion_models", "只含扩散主干，没有 VAE、没有文本编码器"

    if starts("double_blocks", "single_blocks"):
        return "diffusion_models", "Flux 主干（double_blocks / single_blocks）"

    if has("cap_embedder", "noise_refiner", "context_refiner"):
        return "diffusion_models", "DiT 主干（cap_embedder / noise_refiner 结构）"

    if starts("layers.") and has("proj.qkv", "text_embedder.", "x_embedder", "t_embedder",
                                "patch_embed", "adaln"):
        return "diffusion_models", "DiT 主干（layers + 投影/嵌入结构，如 MOSS-Image）"

    # Krea / Chroma 类：根级 blocks.N.attn.wq/wk/wv/wo + mlp
    if starts("blocks.") and has("attn.wq", "attn.wk", "attn.wv", "mlp.gate"):
        return "diffusion_models", "DiT 主干（blocks.N.attn.wq/wk/wv + mlp 结构，如 Krea / Chroma）"

    enc = starts("encoder.down", "first_stage_model.encoder")
    dec = starts("decoder.up", "first_stage_model.decoder")
    if enc and dec:
        return "vae", "含编码器下采样 + 解码器上采样 → VAE"

    if starts("model.language_model.", "model.embed_tokens."):
        return "text_encoders", "LLM / VLM 结构文本编码器（model.language_model / embed_tokens）"

    if starts("model.layers.") and starts("model.embed_tokens"):
        return "text_encoders", "LLM 结构文本编码器（model.layers + embed_tokens，如 Qwen）"
    if starts("text_model.", "text_encoder.", "cond_stage_model.transformer", "encoder.block."):
        return "text_encoders", "CLIP / T5 文本编码器结构"

    if has("control_model", "input_hint_block", "controlnet_cond_embedding", "zero_convs"):
        return "controlnet", "含 ControlNet 专用分支（input_hint_block / zero_convs）"

    if starts("vision_model.", "visual."):
        return "clip_vision", "含视觉编码器结构（vision_model / visual.transformer）"

    if has("emb_params", "string_to_token", "string_to_param"):
        return "embeddings", "Textual Inversion 嵌入（emb_params / string_to_token）"

    return None, None


def detect_from_name(name: str):
    """档名规则。只在读不了内容时用，或者内容判断不出来时兜底。"""
    n = Path(name).name.lower()
    if n.endswith(".gguf"):
        if any(k in n for k in ("t5", "clip", "text_encoder", "encoder")):
            return "text_encoders", "GGUF 文本编码器（按文件名判断）"
        return "diffusion_models", "GGUF 量化主干（按文件名判断）"
    if "audio" in n and "vae" in n:
        return "audio_encoders", "音频 VAE（按文件名判断）"
    if "lora" in n:
        return "loras", "文件名含 lora"
    if "clip_vision" in n:
        return "clip_vision", "文件名含 clip_vision"
    if "controlnet" in n or n.endswith("_t2i.safetensors"):
        return "controlnet", "文件名含 controlnet"
    if "vae" in n:
        return "vae", "文件名含 vae"
    if any(k in n for k in ("t5", "clip_l", "clip_g", "text_encoder", "umt5", "byt5")):
        return "text_encoders", "文件名是文本编码器"
    if any(k in n for k in ("esrgan", "swinir", "upscale", "4x", "2x", "8x")):
        return "upscale_models", "文件名是放大模型"
    if "embedding" in n or "textual" in n:
        return "embeddings", "文件名是 embedding"
    return None, None


def detect(path: Path):
    """返回 dict: folder, reason, tensors, source(content/name/file)。"""
    suffix = path.suffix.lower()

    if suffix in (".safetensors", ".sft"):
        keys = read_header_keys(path)
        if not keys:
            return {
                "folder": None,
                "reason": "文件头读不出来：可能没下载完整，或者根本不是模型文件",
                "tensors": 0,
                "source": "content",
            }
        folder, reason = detect_from_keys(keys)
        if folder:
            return {"folder": folder, "reason": reason, "tensors": len(keys), "source": "content"}
        folder, reason = detect_from_name(path.name)
        return {
            "folder": folder,
            "reason": reason or "头部结构不常见，按文件名也判断不出来",
            "tensors": len(keys),
            "source": "name",
        }

    folder, reason = detect_from_name(path.name)
    return {
        "folder": folder,
        "reason": reason or "这类格式读不了内容，按文件名也判断不出来",
        "tensors": 0,
        "source": "name",
    }


# ---------------------------------------------------------------- 归位

def place_file(src: Path, models_root: Path, folder: str, dry_run=False):
    """移进 models_root/folder，绝不覆盖。返回 dict。"""
    target_dir = models_root / folder
    target = target_dir / src.name

    if target.exists():
        if target.stat().st_size == src.stat().st_size:
            return {"path": str(target), "note": "已存在同名同大小文件，保持原样", "moved": False}
        i = 1
        while True:
            alt = target_dir / f"{src.stem}_{i}{src.suffix}"
            if not alt.exists():
                if not dry_run:
                    target_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(alt))
                return {"path": str(alt), "note": f"已有同名但大小不同，另存为 {alt.name}" + ("（dry-run 未执行）" if dry_run else ""), "moved": not dry_run}
            i += 1

    if not dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(target))
    return {"path": str(target), "note": "将会移动" if dry_run else "已移动", "moved": not dry_run}


# ---------------------------------------------------------------- 输入展开

def expand(paths):
    """把文件/目录展开成待处理的文件列表；跳过临时文件和明显的非模型文件。"""
    out, skipped = [], []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if not f.is_file():
                    continue
                if f.suffix.lower() in TEMP_SUFFIXES:
                    skipped.append((str(f), "临时文件"))
                elif f.suffix.lower() not in MODEL_SUFFIXES:
                    skipped.append((str(f), "不是模型后缀"))
                else:
                    out.append(f)
        elif p.is_file():
            if p.suffix.lower() in TEMP_SUFFIXES:
                skipped.append((str(p), "临时文件"))
            else:
                out.append(p)
        else:
            skipped.append((str(p), "路径不存在"))
    return out, skipped


def collect_paths(args):
    """把位置参数和 --list 文件合并。list 文件一行一个路径（可带引号）。

    走清单文件是为了绕开 cmd 对 & 空格 的转义问题——拖拽 bat 用的就是这条路。
    """
    paths = list(getattr(args, "paths", []) or [])
    lst = getattr(args, "list_file", None)
    if lst:
        try:
            with open(lst, encoding="utf-8-sig", errors="replace") as f:
                for line in f:
                    p = line.strip().strip('"').strip()
                    if p:
                        paths.append(p)
        except Exception as e:
            print(f"读不了清单文件 {lst}：{e}", file=sys.stderr)
    return paths


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TB"


# ---------------------------------------------------------------- 命令

def cmd_detect(args):
    paths = collect_paths(args)
    if not paths:
        print("没有给文件。")
        return 1
    files, skipped = expand(paths)
    results = []
    for f in files:
        d = detect(f)
        d["file"] = str(f)
        d["size"] = f.stat().st_size
        d["size_human"] = human(d["size"])
        results.append(d)

    if args.json:
        print(json.dumps({"files": results, "skipped": skipped}, ensure_ascii=False, indent=2))
        return 0

    if not results:
        print("没有可判断的文件。")
    for r in results:
        label = FOLDER_LABELS.get(r["folder"], r["folder"]) if r["folder"] else "判断不出来"
        mark = _mark(bool(r["folder"]))
        print(f"{mark} {Path(r['file']).name}")
        print(f"    {r['size_human']}  张量 {r['tensors']}")
        print(f"    → {r['folder'] or '(不动)'}   {label}")
        print(f"    依据：{r['reason']}")
    for path, why in skipped:
        print(f"- 跳过 {Path(path).name}（{why}）")
    return 0


def cmd_place(args):
    root = Path(args.models_root).expanduser()
    if not root.is_dir():
        print(f"models 目录不存在：{root}", file=sys.stderr)
        return 2

    paths = collect_paths(args)
    if not paths:
        print("没有给文件。")
        return 1
    files, skipped = expand(paths)
    if not files:
        print("没有可处理的文件。")
        for path, why in skipped:
            print(f"- 跳过 {Path(path).name}（{why}）")
        return 1

    rows = []
    moved = skipped_unknown = 0
    for f in files:
        d = detect(f)
        row = {"file": f.name, "folder": d["folder"], "reason": d["reason"]}
        if not d["folder"]:
            row["note"] = "判断不出来，原样不动"
            skipped_unknown += 1
        else:
            r = place_file(f, root, d["folder"], dry_run=args.dry_run)
            row.update(r)
            if r["moved"]:
                moved += 1
        rows.append(row)

    if args.json:
        print(json.dumps({"placed": rows, "skipped": skipped}, ensure_ascii=False, indent=2))
        return 0

    for r in rows:
        if r["folder"]:
            print(f"{_mark(True)} {r['file']}")
            print(f"    → {r['folder']}  {r['note']}")
            print(f"    {r['path']}")
        else:
            print(f"{_mark(False)} {r['file']}  {r.get('note', '')}")
            print(f"    依据：{r['reason']}")
    for path, why in skipped:
        print(f"- 跳过 {Path(path).name}（{why}）")

    print()
    print(f"共 {len(files)} 个：移动 {moved}，判断不出 {skipped_unknown}，跳过 {len(skipped)}"
          + ("（--dry-run，未真正移动）" if args.dry_run else ""))
    return 0


def cmd_rules(_args):
    print("判定顺序（从上往下，命中即停）：\n")
    for cond, folder in RULES:
        print(f"  {cond}")
        print(f"      → {folder}\n")
    print("读不了头的格式（.ckpt / .pt / .gguf 等）退回文件名规则。")
    print("文件名也判断不出来 → 不动，只报告。")
    return 0


def _utf8_ok():
    enc = (getattr(sys.stdout, "encoding", "") or "").lower()
    return enc in ("utf-8", "utf8")


def _mark(ok):
    """终端不是 UTF-8 时退回 ASCII，免得 GBK 控制台直接抛异常。"""
    if _utf8_ok():
        return "✓" if ok else "?"
    return "[OK]" if ok else "[?]"


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(
        prog="sorter.py",
        description="按内容判断模型类型，并放进 ComfyUI 的 models 子目录。",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect", help="只看这是哪种模型，不动文件")
    d.add_argument("paths", nargs="*")
    d.add_argument("--list", dest="list_file", help="从文件读路径，一行一个（避免 cmd 转义问题）")
    d.add_argument("--json", action="store_true")
    d.set_defaults(func=cmd_detect)

    p = sub.add_parser("place", help="判断并移动到 models/<类型>/")
    p.add_argument("paths", nargs="*")
    p.add_argument("--list", dest="list_file", help="从文件读路径，一行一个（避免 cmd 转义问题）")
    p.add_argument("--models-root", required=True, help="ComfyUI 的 models 目录")
    p.add_argument("--dry-run", action="store_true", help="只报告，不移动")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_place)

    r = sub.add_parser("rules", help="打印判定规则表")
    r.set_defaults(func=cmd_rules)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
