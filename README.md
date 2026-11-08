# ComfyUI-ModelSorter · 模型归位

> 由 @痒局长 制作 · [B 站](https://space.bilibili.com/322892)

> **不用文件名猜。读文件内部的结构，判断它到底是什么模型，放进正确的目录。**

---

## 解决什么问题

你从群里、B 站、迅雷、网盘拿到一个 `.safetensors`，然后卡住：

```
这是大模型？还是 VAE？还是 LoRA？
该放 checkpoints，还是 diffusion_models，还是 loras？
```

**放错了，ComfyUI 里就看不到它；或者更糟——看到了但加载报错。**

网上的说法是「看文件名」：带 `vae` 的放 vae、带 `lora` 的放 loras。

**问题是新模型根本不在文件名里写这些。** 比如：

```
krea2_turbo_fp8_scaled.safetensors     ← 名字里没有 unet/dit 任何线索
moss_image_edit.safetensors            ← 看着像主干，其实是 LoRA
z_image_turbo_bf16.safetensors         ← 名字里看不出是 DiT
```

**这个工具不看名字，直接读文件头里每个张量的名字——那才是结构的指纹。**

---

## One-liner (EN)

ComfyUI model files rarely say what they are in their filename. This tool reads the
`safetensors` header (tensor names only — a few KB, not the weights) and decides whether
a file is a checkpoint, a diffusion backbone, a VAE, a text encoder, a LoRA, a ControlNet,
or an embedding — then moves it into the correct `models/<type>/` folder.

No network. No config. Pure standard library.

---

## 用法

```bash
# 只看这是什么，不动文件
python sorter.py detect 下载的模型.safetensors

# 判断并归位
python sorter.py place  下载的模型.safetensors --models-root <ComfyUI>/models

# 只看会放到哪，不真动
python sorter.py place  某目录 --models-root <ComfyUI>/models --dry-run

# 打印完整规则表
python sorter.py rules
```

**Windows 不想敲命令：把文件或文件夹直接拖到 `拖进去排序.bat` 上。**

> 第一次用之前，改一下 bat 里那行 `set MODELS=...`，指向你自己的 `models` 目录。

---

## 输出长这样

```
✓ krea2_turbo_fp8_scaled.safetensors
    12.24 GB  张量 686
    → diffusion_models   扩散主干（UNet / DiT）
    依据：DiT 主干（blocks.N.attn.wq/wk/wv + mlp 结构，如 Krea / Chroma）
```

**它会告诉你三件事：判成了什么、凭什么、放到哪。**
**判断不出来的，它说「判断不出来」并原样不动——绝不猜着放。**

---

## 判定规则

按顺序，命中即停。全部基于 `safetensors` 头部的张量名：

| 判定条件 | 归属目录 |
|---|---|
| `lora_unet` / `lora_te` / `lora_down.` / `lora_up.` / `lora_a` / `lora_b` | `loras` |
| `model.diffusion_model` + (`vae.` / `first_stage_model` / `cond_stage_model` / `text_encoders.`) | `checkpoints` |
| `model.diffusion_model`（单独出现） | `diffusion_models` |
| `double_blocks` / `single_blocks` | `diffusion_models`（Flux 主干） |
| `cap_embedder` / `noise_refiner` / `context_refiner` | `diffusion_models`（DiT） |
| `layers.` + (`proj.qkv` / `text_embedder.` / `x_embedder` / `t_embedder` / `patch_embed` / `adaln`) | `diffusion_models`（MOSS-Image 类） |
| `blocks.` + (`attn.wq` / `attn.wk` / `attn.wv` / `mlp.gate`) | `diffusion_models`（Krea / Chroma 类） |
| `encoder.down*` 且 `decoder.up*` | `vae` |
| `model.language_model.` / `model.embed_tokens.` | `text_encoders`（LLM / VLM 结构） |
| `model.layers` + `model.embed_tokens` | `text_encoders`（Qwen 类） |
| `text_model.` / `text_encoder.` / `cond_stage_model.transformer` / `encoder.block.` | `text_encoders`（CLIP / T5） |
| `control_model` / `input_hint_block` / `controlnet_cond_embedding` / `zero_convs` | `controlnet` |
| `vision_model.` / `visual.` | `clip_vision` |
| `emb_params` / `string_to_token` / `string_to_param` | `embeddings` |

**读不了头的格式**（`.ckpt` / `.pt` / `.gguf` 等）退回文件名规则；
**文件名也判断不出来 → 不动，只报告。**

---

## 安全设计

这个工具会移动你的文件，所以三条硬规矩：

```
① 绝不覆盖
   同名同大小 → 跳过，保持原样
   同名不同大小 → 另存 xxx_1.safetensors，原文件不动

② 判断不出来就不动
   只打印「判断不出来」，不猜、不乱放

③ --dry-run 先看
   报告会放到哪，一个字节都不动
```

**另外：只读文件头（通常几百 KB），不加载权重，不解析张量内容。**

---

## 实测

本机模型库 15 个真实文件，全部判出，与现有目录比对 **14/15 一致**：

```
✓ flux1-dev-fp8                → checkpoints
✓ krea2_turbo_fp8_scaled       → diffusion_models
✓ krea2_turbo_int8_convrot     → diffusion_models
✓ moss_image_sft_int8_convrot  → diffusion_models
✓ z_image_turbo_bf16           → diffusion_models
✓ krea2_darkbrush              → loras
✓ qwen3vl_4b_fp8_scaled        → text_encoders
✓ qwen3vl_4b_int8_convrot      → text_encoders
✓ qwen3_5_9b                   → text_encoders
✓ qwen3_5_vl_4b                → text_encoders
✓ qwen_3_4b                    → text_encoders
✓ ae                           → vae
✓ moss_ae                      → vae
✓ qwen_image_vae               → vae
✗ moss_image_edit              → 判定 loras，但实际放在非标准目录 moss-image-edit
                                 （按内容确实是 LoRA，应当归入 loras）
```

**那个「不一致」正好说明这个工具干什么用：它按内容判，不按现状判。**

---

## 已知限制（请先读）

```
① 规则是白名单
   全新架构（又一套没人见过的命名）会判不出来。
   判不出来就是判不出来——不会猜。遇到请在 issue 里附上开头几个张量名。

② 老格式只能靠文件名
   .ckpt / .pt / .gguf 读不到张量名，准确率取决于文件名起得好不好。

③ 文件没下完会判不出来
   头部读不全 → 「可能没下载完整」。这是有意的，防止半截文件被归位。

④ 只移动，不下载
   要「缺模型时自动补」，那是另一个项目的事。
```

---

## 免责声明

```
⚠️ 本工具只做一件事：读文件头，判断类型，把文件移动到对应目录。
   不下载、不修改文件内容、不删除任何文件。

⚠️ 移动文件前请确认 --models-root 指向正确。
   建议第一次先用 --dry-run 看一眼。

⚠️ 判断结果仅供参考。放错目录导致的问题由使用者自行承担。

⚠️ 本工具与 ComfyUI / Comfy Org 官方无隶属关系。

⚠️ 使用本工具产生的任何后果由使用者自行承担。
```

---

## 署名

**由 @痒局长 制作。**

- GitHub：[AinJuZen](https://github.com/AinJuZen)
- B 站：[痒局长](https://space.bilibili.com/322892)

问题、建议、新架构的规则，都欢迎开 issue。

---

## License

MIT — 见 [LICENSE](LICENSE)。
