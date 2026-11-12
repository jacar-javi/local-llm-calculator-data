# models.json: research notes (2026-09-15)

46 entries: the 20 existing ids re-verified, plus 26 new. Every number was read on the research date from a Hugging Face `config.json`, the HF model API (`safetensors.total`, `createdAt`), safetensors shard headers (HTTP Range), the model card, the transformers source for that architecture, or the vendor's announcement page. Raw files are in `calc/raw/`, and `calc/build.py` regenerates `models.json` from them.

## Fields added beyond the requested schema

| Field | Meaning |
|---|---|
| `kv_heads_full`, `head_dim_full` | KV heads and head dim on the full-attention layers. They differ from `kv_heads`/`head_dim` only on Gemma 4. |
| `kv_elements_per_token_full` | Cache elements per token summed over the full layers: 2 x layers x kv_heads_full x head_dim_full. For MLA it is layers x (kv_lora_rank + qk_rope_head_dim). |
| `kv_elements_per_token_swa` | Same sum over the SWA or chunked layers. Multiply by min(ctx, swa_window). |
| `state_layers`, `fixed_state_elements` | Gated DeltaNet or Mamba2 layers and their total recurrent + conv state. This state does not grow with context. |
| `max_context_extended` | Officially documented extended context (YaRN etc.), or null. |
| `released_source` | Where the release date came from: `blog`, `card`, or `hf_createdAt`. `hf_createdAt` is the repo creation date and may predate the public release. |

**Conventions.** `kv_layers_swa` is 0 and `swa_window` is null when a model has no windowed layers. On MLA models `kv_heads`, `head_dim`, `kv_heads_full` and `head_dim_full` are null on purpose: the fields do not apply, the values were not unverifiable. `max_context` is the context the card states; when the card gives none, it is the config's `max_position_embeddings`.

**Suggested formula.** KV bytes = (`kv_elements_per_token_full` x ctx + `kv_elements_per_token_swa` x min(ctx, `swa_window`)) x bytes_per_element, plus `fixed_state_elements` x state bytes. Nemotron's config sets `mamba_ssm_cache_dtype` to float32.

## Models skipped, and why

Every requested model exists with downloadable weights. None turned out to be API-only or missing.

- **DeepSeek V4 (V4-Flash, V4-Pro, V4.1-Flash): not added.** V4 does not use MLA: it has no `kv_lora_rank`. Each layer has a shared-KV MQA sliding branch (1 KV head x 512, window 128) plus a compressed branch. `compress_ratios` gives 2 sliding-only layers, 21 CSA layers (1/4 compression, plus an indexer with `index_head_dim` 128) and 20 HCA layers (1/128). See transformers `deepseek_v4/configuration_deepseek_v4.py` and its `DeepseekV4CSACache`/`HCACache`. That fits none of the five attention types, so **DeepSeek-V3.1** (pure MLA) is the entry. Adding V4 later would need a new attention type modelled on `modeling_deepseek_v4.py`. I did not derive a per-token formula for it.
- **DeepSeek V3.2: not added.** It is MLA plus a DSA indexer key cache: `past_key_values.update_indexer`, `index_head_dim` 128. transformers also notes that "sparse-attention models cache the expanded K/V". The plain MLA formula would undercount it.
- **GLM-5.3-Flash: not used as the GLM pick.** It is 321B (FP8) and mixes linear attention with DeepSeek sparse attention. **GLM-4.7-Flash** (31B, MLA) is included as the newest locally runnable Flash, alongside GLM-4.5-Air.
- **Kimi K2.6 and Kimi K3: not added.** K2.6 is the K2 text architecture plus vision, with INT4 pack-quantized I32 weights. K3 is 2.78T parameters on a Kimi-Linear hybrid. Kimi-K2-Instruct-0905 (FP8, 256K) is the K2 entry.
- **Nemotron 3.5 Lightning 30B-A3B: not used.** Its HF API `safetensors.total` (31,577,937,344) does not equal the sum of its own per-dtype counts (32,913,266,240). **Nemotron 3 Nano 30B-A3B** is used instead: its API total is consistent, and it has a card release date.
- **Granite 4.1 / 4.2: not used.** The `granite-4.2-30b` and `-8b` configs are dense `GraniteForCausalLM` with no Mamba layers. Granite-4.0-H-Small stays as the hybrid.
- **Not requested, so not added:** Qwen3.8-Flash-Next (180B, qwen-community license), Mistral Small 4 119B, Qwen3.5-27B/35B/122B/397B and Qwen3.6-27B. Qwen3.6-27B has the same architecture as the included Qwen3.8-27B.
- **Qwen3.5 small dense sizes:** 0.8B, 2B, 4B and 9B all exist, and all are included.

## Ambiguous fields and how I resolved them

1. **Gated repos** (google/gemma-3, meta-llama, mistralai Large 2407). Their `config.json` returns 401. The HF tree API still lists each file's git blob SHA-1. The `unsloth/*` mirrors have byte-identical safetensors shards, but their config.json files are re-serialized and **not** byte-identical. I found mirrors whose config.json SHA-1 matches the official one and recomputed that hash locally:

   | Model | Mirror with byte-identical config.json |
   |---|---|
   | Gemma 3 1B | rapacious/gemma-3-1b-it |
   | Gemma 3 4B, 12B, 27B | second-state/gemma-3-{4b,12b,27b}-it-GGUF |
   | Llama 3.1 8B | SillyTilly/Meta-Llama-3.1-8B-Instruct |
   | Llama 3.3 70B | yejingfu/Llama-3.3-70B-Instruct |
   | Llama 4 Scout, Maverick | RedHatAI/Llama-4-{Scout,Maverick}-... |
   | Mistral Large 2407 | second-state/Mistral-Large-Instruct-2407-GGUF |

   `params_b` for these models comes from the official repo's API, which works without auth.
2. **Gemma 3 official configs leave fields out.** The 4B config has no heads, KV heads, head_dim or pattern, and the 12B config has no head_dim. Gemma3TextConfig defaults fill them: 8 heads, 4 KV heads, head_dim 256, pattern 6. The layer split comes from `layer_types = full iff (i+1) % 6 == 0`: 1B = 4 full + 22 SWA (window 512), 4B = 5 + 29, 12B = 8 + 40, 27B = 10 + 52 (window 1024).
3. **Gemma 4.**
   - Full layers use `global_head_dim` (512). They use `num_global_key_value_heads` only when `attention_k_eq_v` is true: 31B has 4, 26B has 2, 12B has 1. E2B and E4B have k_eq_v false, so they keep `num_key_value_heads`.
   - k_eq_v shares weights, not cache: K and V are normed differently and both are cached.
   - E2B and E4B set `num_kv_shared_layers` (20 and 18). Those trailing layers own no cache (`cache_utils` drops them). That leaves E2B with 3 full + 12 SWA and E4B with 4 full + 20 SWA.
   - **The task template's gemma4-31b example is wrong in three places.** Its `head_dim` 128 should be 256 on SWA layers and 512 on full layers, with 4 KV heads on the full layers. Its `released` 2026-03-31 should be 2026-04-02 (launch blog). Its `params_b` 30.7 is the card's figure without the ~0.58B vision tower; the HF total is 31.3.
4. **Llama 4.** Scout's official config has `no_rope_layers: []`, and Maverick's has no such key. transformers treats both as the default: a NoPE global layer every 4th layer, 12 in all. The other 36 layers are "chunked_attention", cached as a sliding layer of `attention_chunk_size` = 8192. `kv_layers_swa`/`swa_window` hold the chunked layers.
5. **Hybrid KV counts.**
   - Qwen3.5, 3.6 and 3.8 count `layer_types == full_attention`, so 1 in 4 layers carries KV.
   - Granite counts `layer_types == attention` (4 of 40).
   - Nemotron-H counts `*` in `hybrid_override_pattern` (6 of 52 blocks; M = Mamba2, E = MoE).
   - Recurrent state uses the shapes in the transformers modeling files. The conv state is counted as (kernel − 1) columns; transformers may keep `kernel` columns, a negligible difference.
6. **MLA.** transformers `deepseek_v3` and `glm4_moe_lite` both cache the compressed latent ("Cache read / write is performed while latent KV is still compressed"). That confirms layers x (kv_lora_rank + qk_rope_head_dim).
7. **MTP layers are in the params but not in the KV counts.** GLM-4.5-Air `layers.46`, GLM-4.7-Flash `layers.47`, DeepSeek-V3.1 (13.5B) and the Qwen3.5/3.6/3.8 `mtp.*` weights all appear in `params_b`. None of them is counted in `kv_layers_full`.
8. **Context disagreements.**

   | Model | Card | Config | Recorded |
   |---|---|---|---|
   | Qwen3 | 32,768 native, 131,072 with YaRN | 40,960 (32,768 output + 8,192 prompt, per card) | 32,768 |
   | DeepSeek-V3.1 | 128K | 163,840 | 131,072 |
   | Devstral Small 2 | "256k" | 393,216 | 262,144 |
   | Nemotron 3 Nano | up to 1M | 262,144 (VRAM default) | 1,048,576 |
   | SmolLM3 | 64k trained, 128k via YaRN | 65,536 | 65,536 (+ extended) |
   | gpt-oss, GLM-4.5-Air, GLM-4.7-Flash | not stated | 131,072 / 131,072 / 202,752 | config |

9. **Windows larger than the context.** Phi-4-mini sets `sliding_window` 262,144, above its 131,072 max, so it is effectively full attention. R1-Distill-Qwen-32B sets `sliding_window` 131,072 with `use_sliding_window` false, so it is full attention.
10. **gpt-oss params and bits.** The MoE weights are MXFP4 (U8 blocks + scales). The API `safetensors.total` already counts 2 params per block byte, and it matches my header-derived count exactly: 20,914,757,184 and 116,829,156,672. `native_bits` = 4 for the experts (4.25 bits with scales); the rest is BF16.
11. **Maple-Preview.** The card says "ternary-weight", but the HF transformers checkpoint is BF16 (20.21B params, ~40.4 GB). modeling_maple.py contains no quantization code. `native_bits` = 1.58 (log2 3). The card's "5.31 GB checkpoint" equals the 2-bit MLX repo (5,314,328,134 bytes). The official GGUFs are TQ1_0 at 4.64/5.06 GiB and TQ2_0 at 5.50/5.91 GiB.
12. **FP8-native checkpoints** (`native_bits` 8): DeepSeek-V3.1, Kimi-K2-0905 and Devstral Small 2. For DeepSeek and Kimi the API total includes F32 scale tensors (41.5M and 62.5M), which is immaterial.
13. **params_b vs model name.** The HF total counts vision towers and MTP layers, so it can differ from the name:

    | Model | HF total (B) | Card or name figure |
    |---|---|---|
    | Qwen3.5 0.8B | 0.9 | 0.8 (text 0.75) |
    | Qwen3.5 2B | 2.3 | 2 |
    | Qwen3.5 4B | 4.7 | 4 |
    | Qwen3.5 9B | 9.7 | 9 |
    | Qwen3.6 35B-A3B | 36.0 | 35 |
    | Gemma 4 E2B | 5.1 | 2.3 effective |
    | Gemma 4 E4B | 8.0 | 4.5 effective |
    | Gemma 4 26B A4B | 25.8 | 25.2 |
    | Gemma 4 31B | 31.3 | 30.7 |
    | GLM-4.5-Air | 110.5 | 106 |
    | GLM-4.7-Flash | 31.2 | 30B |
    | DeepSeek-V3.1 | 684.5 | 671 |
    | Kimi K2 | 1026.5 | 1T |
    | Nemotron 3 Nano | 31.6 | 30 |
    | Granite 4.0 H Small | 32.2 | 32 |
    | Llama 4 Maverick | 401.6 | 400 |
    | Llama 4 Scout | 108.6 | 109 |

    Each entry's notes give the text/vision/MTP split from the shard headers.
14. **Release dates.** Dates come from a blog or the card where one was readable. Several announcement pages (qwen.ai, z.ai, the Kimi site) render with JavaScript and gave no date, so these entries fall back to `hf_createdAt`: qwen36-35b, the four qwen35 sizes, qwen38-27b, mistral-s32, glm45-air, glm47-flash, kimi-k2, minicpm5-2b, spark-x25-4b and maple-preview.

## Existing entries whose previous values were wrong

Old values are the hand-written `MODELS` array of calculator v1, before the data moved into these JSON files.

The KV figures below are KV cache at 32k context in fp16, computed with the old flat formula and with the verified split.

| id | old | verified | KV @32k fp16 |
|---|---|---|---|
| gemma3-1b | kvLayers 26 | 4 full + 22 SWA (window 512) | 0.81 → 0.14 GiB |
| gemma3-4b | params 4, kvLayers 34 | 4.3; 5 full + 29 SWA (1024) | 4.25 → 0.74 GiB |
| gemma3-12b | params 12, kvLayers 48 | 12.2; 8 full + 40 SWA (1024) | 12.00 → 2.31 GiB |
| gemma3-27b | params 27, kvLayers 62 | 27.4; 10 full + 52 SWA (1024) | 15.50 → 2.91 GiB |
| llama4scout | params 109, kvLayers 48 | 108.6; 12 full + 36 chunked (8192) | 6.00 → 2.62 GiB |
| qwen36-35b | params 35 | 36.0 (incl. 0.84B MTP + 0.45B vision); layers/KV were right | 0.62 GiB, plus 16.5M state elements not modelled before |
| nemo12b / phi4 | params 12 / 14 | 12.2 / 14.7 | unchanged |
| mistral-l2 | params 123 | 122.6 (card still says 123B) | unchanged |

Layers, hidden size, heads, KV heads and head_dim were correct for all 20 old entries. That includes the explicit-head_dim traps (Qwen3, Nemo, Mistral Small, Gemma 3 27B) and the Qwen3.6 KV-layer count of 10. Unchanged entries: qwen3-4b, qwen3-8b, qwen3-14b, qwen3-32b, qwen3-30b, qwen3-235b, llama8b, llama33-70b, ds-r1-8b, ds-r1-32b and mistral-s31.

## 2026-11-12 update: new chat models, decision models, voice models

**Chat models (`models.json`).** 14 entries added and Spark-X2.5 4B replaced by Spark-X2.6 4B (same config, so the same memory). Same method as above: `config.json`, HF API, model card, modeling source. Points to know:

- LFM3 and LFM2.5 keep a short-convolution state on their non-attention layers. It is counted in `fixed_state_elements` like the Mamba/DeltaNet conv state (hidden x (kernel - 1)); the attention label (`swa-hybrid` for LFM3, `linear-hybrid` for LFM2.5) is the closest of the five.
- MiMo-V2.6-Flash has different K and V head sizes (192 / 128), so its KV elements are layers x kv_heads x 320, not 2 x heads x head_dim. Do not recompute them from `kv_heads`/`head_dim`.
- Xing4.0 is counted with the MLA latent (40 x 576), as on the other MLA entries; its own `trust_remote_code` modeling caches the decompressed K/V (409,600 per token), so a plain transformers run uses far more.
- 12 of the 15 release dates are the HF repo creation date (`released_source`).
- Skipped: DeepSeek-V4.1-Flash (compressed attention), MiniMax-M3.1-Flash and Step-5-Preview (index-cache sparse attention), Ling-3.1-flash and Raon-1 (MLA mixed with linear layers), Nemotron Dory (looped), fine-tunes of architectures already listed, and models above ~800B.

**Decision models (`decision-models.json`).** 14 models that return a probability per allowed option in one forward pass (Clef, Clef-flash, Clef-omni, OpenJev, OpenJev Flash, d3/-flash/-mini, pplx-decider v1.2/-flash, Kev-27B, Nimble-v4-max, H2O Lightning 4B, Liquid d1-3B). Each has a full `models.json`-schema entry plus a `decision` block (base model, output read-out, runtimes, formats, published throughput, licence). Memory uses the same weights + KV formula; the generator adds Clef's separately shipped joint head (0.13B / 0.12B). Per-engine support (`yes`/`helper`/`no`/undocumented) and the translated speed figures are editorial and live in `DECISION_RUNTIMES` / `DECISION_REFS` in `scripts/build-llm-calc-data.py`. No decode or prefill estimate: prefill is compute bound and not calibrated.

**Voice models (`tts-models.json`).** 39 open TTS models with components, languages, cloning, streaming, licence, weights per format, runtimes and backends, published memory and published speed (RTF, x real time, time to first audio), each with its source. The Open TTS Leaderboard tables were read on the same day. Memory in the calculator is the smallest published figure that applies to the platform kind (`TTS_MEM` in the generator); without one, the 16-bit weights stand in and are flagged as a minimum. Spanish support and commercial-use conditions are summarised in `TTS_SPANISH` / `TTS_COMMERCIAL_NOTE`.
