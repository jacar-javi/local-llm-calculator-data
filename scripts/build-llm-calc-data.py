#!/usr/bin/env python3
"""Generate the data block of the local LLM calculator from verified JSON.

Sources (platforms and engines verified on 2026-09-15; models, decision and voice models on
2026-11-12; every value with its URL):
  data/models.json           model architectures (config.json, HF API)
  data/hardware.json         platforms + llama.cpp speed calibration (dense models)
  data/moe-calibration.json  speed calibration on MoE models
  data/engines.json          engines pinned to a release tag
  data/colibri.json          Colibri expert-streaming catalogue
  data/decision-models.json  decision models (LLM-shaped) + runtimes
  data/tts-models.json       voice models: published memory and speed

Output: replaces everything between `/* DATA:BEGIN */` and `/* DATA:END */` in
calculator/llm-calculator.js.

This script adds only EDITORIAL choices on top of the verified facts, all visible
below: which formats/KV types each engine offers in the UI, the bilingual notes,
the RAM variants a platform is expanded into, and three labelled assumptions
(4 GiB kept for the OS on CPU-only machines, each engine's flat overhead, SGLang's
Metal runtime left out as undocumented in its hardware index).

Usage: python3 scripts/build-llm-calc-data.py [--check]
  --check  exit 1 if the JS block is not what this script would write.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
JS = ROOT / "calculator" / "llm-calculator.js"
VERIFIED = "2026-11-12"


def yes(v) -> bool:
    return isinstance(v, str) and v.lower().startswith("yes")


# ------------------------------------------------------------------ engines
# formats / kv: ids from engines.json in the order offered. overheadGiB: flat engine
# overhead added to weights + KV; each value states what it approximates.
ENGINE_UI = {
    "llamacpp": {
        "formats": ["gguf-bf16", "gguf-q8_0", "gguf-q6_k", "gguf-q5_k_m", "gguf-q4_k_m", "gguf-q4_0",
                    "gguf-iq4_xs", "gguf-q3_k_m", "gguf-iq3_xxs", "gguf-q2_k", "gguf-tq2_0", "gguf-tq1_0"],
        "default": "gguf-q4_k_m", "kv": ["f16", "q8_0", "q5_1", "q4_0"],
        "overheadGiB": 1.0, "swaPadTokens": 512,
        "note": ("Reserva la caché KV del contexto completo al cargar. llama-server comparte una sola caché entre sus ranuras: "
                 "para atender N peticiones de C tokens a la vez hace falta un contexto de N×C. En las capas de ventana deslizante "
                 "guarda solo la ventana más un micro-lote de 512 tokens.",
                 "Allocates the KV cache for the whole context at load. llama-server shares one cache across its slots: serving "
                 "N requests of C tokens at once needs an N×C context. Sliding-window layers keep only the window plus one "
                 "512-token micro-batch."),
        "policyNote": ("Margen ≈1 GB: búferes de cálculo (898 MiB en el ejemplo documentado con micro-lote de 512).",
                       "Overhead ≈1 GB: compute buffers (898 MiB in the documented example at a 512 micro-batch)."),
    },
    "ollama": {
        "formats": ["gguf-bf16", "gguf-q8_0", "gguf-q6_k", "gguf-q5_k_m", "gguf-q4_k_m", "gguf-q4_0",
                    "gguf-iq4_xs", "gguf-q3_k_m", "gguf-q2_k"],
        "default": "gguf-q4_k_m", "kv": ["f16", "q8_0", "q4_0"],
        "overheadGiB": 1.5, "swaPadTokens": 512,
        "note": ("Lanza llama-server con num_ctx × OLLAMA_NUM_PARALLEL, así que la caché se multiplica por las peticiones simultáneas "
                 "(con Qwen3.5/3.6 y Nemotron-H fuerza una). Si no fijas el contexto, depende de la VRAM total: 4.096 tokens por debajo "
                 "de 23 GiB, 32.768 hasta 47 GiB y 262.144 por encima.",
                 "Starts llama-server with num_ctx × OLLAMA_NUM_PARALLEL, so the cache scales with concurrent requests (Qwen3.5/3.6 "
                 "and Nemotron-H are forced to one). If you do not set the context it depends on total VRAM: 4,096 tokens below "
                 "23 GiB, 32,768 up to 47 GiB and 262,144 above."),
        "policyNote": ("Margen ≈1,5 GB: reserva mínima por GPU (457 MiB, 512 MiB en Metal) más los búferes de cálculo.",
                       "Overhead ≈1.5 GB: minimum per-GPU reserve (457 MiB, 512 MiB on Metal) plus compute buffers."),
    },
    "lmstudio": {
        "formats_from": {"llamacpp": ["gguf-q8_0", "gguf-q6_k", "gguf-q5_k_m", "gguf-q4_k_m", "gguf-iq4_xs", "gguf-q3_k_m"]},
        "formats": ["lmstudio-mlx-8bit-g64", "lmstudio-mlx-4bit-g64"],
        "default": "gguf-q4_k_m", "kv": ["f16", "q8_0", "q4_0", "mlx-kv-8bit-g64", "mlx-kv-4bit-g64"],
        "overheadGiB": 1.0, "swaPadTokens": 0,
        "note": ("Motor llama.cpp para GGUF y MLX en Mac. Es código cerrado: no documenta si guarda solo la ventana deslizante ni qué "
                 "arquitecturas híbridas soporta, así que la caché se calcula con el contexto completo. Por defecto comparte una "
                 "caché entre 4 predicciones.",
                 "llama.cpp engine for GGUF and MLX on Macs. It is closed source: it does not document whether it stores only the "
                 "sliding window or which hybrid architectures it supports, so the cache is costed at full context. By default it "
                 "shares one cache across 4 predictions."),
        "policyNote": ("Margen ≈1 GB, como el motor llama.cpp que incluye.", "Overhead ≈1 GB, like the llama.cpp engine it bundles."),
    },
    "vllm": {
        "formats": ["bf16", "fp8-w8", "awq-w4-g128", "gptq-w4-g128", "ct-w4a16-g128", "nvfp4"],
        "default": "awq-w4-g128", "kv": ["auto", "fp8", "nvfp4"], "backends_drop": ["Metal"],
        "overheadGiB": 1.5, "swaPadTokens": 0,
        "note": ("Reserva al arrancar el 92 % de la memoria de cada GPU (gpu_memory_utilization en el código de v0.29.0; la "
                 "documentación aún dice 0,9). Ese bloque cubre pesos, activaciones y grafos CUDA, y lo que sobra es una caché KV "
                 "paginada que comparten todas las peticiones. En Mac solo existe un plugin comunitario, que no se cuenta.",
                 "At start it reserves 92% of each GPU's memory (gpu_memory_utilization in the v0.29.0 code; the docs still say "
                 "0.9). That block covers weights, activations and CUDA graphs, and the rest becomes a paged KV cache shared by all "
                 "requests. On Macs only a community plugin exists, which is not counted."),
        "policyNote": ("Margen ≈1,5 GB para activaciones y grafos CUDA, que vLLM mide al arrancar.",
                       "Overhead ≈1.5 GB for activations and CUDA graphs, which vLLM profiles at start."),
    },
    "sglang": {
        "formats": ["bf16", "fp8-w8", "awq-w4-g128", "gptq-w4-g128", "nvfp4"],
        "default": "awq-w4-g128", "kv": ["auto", "fp8_e4m3", "nvfp4"],
        "backends_drop": ["Metal", "CPU"], "policy": {"type": "sglang"}, "swaWindowOnly": False,
        "overheadGiB": 0.0, "swaPadTokens": 0,
        "note": ("Reserva por GPU 512 MB + 1,5 × el tamaño del prefill por bloques + búferes de grafos (fórmula de v0.5.19). Su caché "
                 "de ventana deslizante es proporcional (80 % de la global), no un tope, así que se calcula con el contexto completo. "
                 "No se cuentan su runtime MLX para Mac ni su backend de CPU, que exige Xeon con AMX.",
                 "Reserves per GPU 512 MB + 1.5 × the chunked-prefill size + graph buffers (v0.5.19 formula). Its sliding-window "
                 "cache is proportional (80% of the global one), not a cap, so it is costed at full context. Its Mac MLX runtime "
                 "and its CPU backend, which needs Xeon with AMX, are not counted."),
        "policyNote": ("La reserva de SGLang ya incluye activaciones; no se suma margen aparte.",
                       "SGLang's reserve already covers activations; no separate overhead is added."),
    },
    "mlx-lm": {
        "formats": ["bf16", "mlx-8bit-g64", "mlx-6bit-g64", "mlx-5bit-g64", "mlx-4bit-g64", "mlx-4bit-g32",
                    "mlx-4bit-g128", "mlx-mxfp4", "mlx-3bit-g64", "mlx-2bit-g64"],
        "default": "mlx-4bit-g64", "kv": ["fp16", "q8-g64", "q6-g64", "q4-g64"],
        "overheadGiB": 0.5, "swaPadTokens": 0, "policy": {"type": "on-demand"},
        "note": ("La caché crece bajo demanda en pasos de 256 tokens; aquí se calcula con el contexto lleno. En Mac sube el límite de "
                 "memoria de la GPU al máximo recomendado de Metal. También funciona con CUDA en Linux.",
                 "The cache grows on demand in 256-token steps; this costs it at full context. On Macs it raises the GPU wired "
                 "limit to Metal's recommended maximum. It also runs on CUDA under Linux."),
        "policyNote": ("Margen ≈0,5 GB (supuesto: MLX no documenta una cifra).", "Overhead ≈0.5 GB (assumption: MLX documents no figure)."),
    },
    "omlx": {
        "formats": ["oq8", "oq6", "oq5", "oq4", "oq3.5", "oq3", "oq2"],
        "formats_from": {"mlx-lm": ["mlx-8bit-g64", "mlx-4bit-g64"]},
        "default": "oq4", "kv": ["fp16"],
        "kv_extra": [
            {"id": "tq8", "label": "TurboQuant 8 bits (≈)", "bytes": 8 / 8 + 2 / 128},
            {"id": "tq4", "label": "TurboQuant 4 bits (≈)", "bytes": 4 / 8 + 2 / 128},
            {"id": "tq3", "label": "TurboQuant 3 bits (≈)", "bytes": 3 / 8 + 2 / 128},
        ],
        "policy": {"type": "ceiling", "reserveGiB": 6, "smallReserveGiB": 4, "smallBelowGiB": 24},
        "overheadGiB": 0.5, "swaPadTokens": 0,
        "note": ("Solo Apple Silicon. Tope de memoria = el menor entre la RAM menos 6 GiB (4 GiB en equipos de menos de 24 GiB, perfil "
                 "equilibrado por defecto) y el límite de Metal. La caché en SSD guarda bloques expulsados; no reduce la caché de las "
                 "peticiones activas. TurboQuant se estima para head_dim 128.",
                 "Apple Silicon only. Memory ceiling = the lower of RAM minus 6 GiB (4 GiB on machines under 24 GiB, default "
                 "balanced tier) and Metal's limit. The SSD cache holds evicted blocks; it does not shrink the cache of active "
                 "requests. TurboQuant is estimated for head_dim 128."),
        "policyNote": ("Margen ≈0,5 GB: la holgura de 512 MiB que reserva oMLX.", "Overhead ≈0.5 GB: the 512 MiB slack oMLX reserves."),
    },
    "exllamav3": {
        "formats_nominal": [8.0, 6.0, 5.0, 4.0, 3.5, 3.0, 2.5],
        "default": "exl3-4.0", "kv": ["fp16", "q8", "q6", "q4"],
        "overheadGiB": 1.0, "swaPadTokens": 0,
        "note": ("Solo NVIDIA. EXL3 cuantiza a la tasa media que eliges al convertir (convert.py -b), así que los formatos son "
                 "objetivos nominales; con -hq el tamaño real sube un poco. Reserva max_num_tokens compartido por todas las peticiones.",
                 "NVIDIA only. EXL3 quantizes to the average bitrate you pick at conversion (convert.py -b), so these formats are "
                 "nominal targets; with -hq the real size is slightly larger. It reserves max_num_tokens shared by all requests."),
        "policyNote": ("Margen ≈1 GB (supuesto).", "Overhead ≈1 GB (assumption)."),
    },
}

BACKEND_KEYS = ["CUDA", "ROCm", "Vulkan", "Metal", "SYCL"]
# Readable KV cache labels (the browser appends bytes per element in the page language).
KV_LABELS = {
    "f16": "F16", "q8_0": "Q8_0", "q5_1": "Q5_1", "q4_0": "Q4_0",
    "mlx-kv-8bit-g64": "MLX 8 bits", "mlx-kv-4bit-g64": "MLX 4 bits",
    "auto": "FP16/BF16", "fp8": "FP8", "fp8_e4m3": "FP8", "nvfp4": "NVFP4",
    "fp16": "FP16", "q8-g64": "8 bits", "q6-g64": "6 bits", "q4-g64": "4 bits",
    "q8": "8 bits", "q6": "6 bits", "q4": "4 bits",
}
POLICY_MAP = {"ctx-prealloc": "ctx-prealloc", "on-demand": "on-demand", "fraction-prealloc": "fraction", "ceiling": "ceiling"}


def build_engines(raw: dict) -> list[dict]:
    by_id = {e["id"]: e for e in raw["engines"]}
    out = []
    for eid, ui in ENGINE_UI.items():
        e = by_id[eid]
        backends = [b for b in BACKEND_KEYS if yes(e["backends"].get(b))]
        if yes(e["backends"].get("CPU-x86")):
            backends.append("CPU")
        backends = [b for b in backends if b not in ui.get("backends_drop", [])]
        fmts = {f["id"]: f for f in e["weight_formats"]}
        formats = []
        for src, ids in ui.get("formats_from", {}).items():
            src_f = {f["id"]: f for f in by_id[src]["weight_formats"]}
            formats += [{"id": i, "label": src_f[i]["label"], "bpw": src_f[i]["bpw"]} for i in ids]
        for i in ui.get("formats", []):
            if fmts[i]["bpw"] is None:
                raise SystemExit(f"{eid}: format {i} has no verified bpw")
            formats.append({"id": i, "label": fmts[i]["label"], "bpw": fmts[i]["bpw"]})
        for bpw in ui.get("formats_nominal", []):
            formats.append({"id": f"exl3-{bpw:.1f}", "label": f"EXL3 {bpw:.1f} bpw (objetivo)", "bpw": bpw})
        kvs = {k["id"]: k for k in e["kv_cache_dtypes"]}
        kv = [{"id": i, "label": KV_LABELS.get(i, i), "bytes": kvs[i]["bytes"]} for i in ui["kv"]]
        kv += ui.get("kv_extra", [])
        att = e["attention_support"]
        pol = e["memory_policy"]
        policy = ui.get("policy") or {"type": POLICY_MAP[pol["type"]]}
        if policy["type"] == "fraction":
            policy["fraction"] = pol["fraction"]
        out.append({
            "id": eid, "name": e["name"], "version": e["version"], "backends": backends,
            "formats": formats, "defaultFormat": ui["default"], "kvDtypes": kv,
            "policy": policy,
            "swaWindowOnly": ui.get("swaWindowOnly", att.get("swa_window_only")),
            "swaPadTokens": ui["swaPadTokens"], "mla": att.get("mla"),
            "linearHybrid": att.get("linear_hybrid"), "ssmHybrid": att.get("ssm_hybrid"),
            "overheadGiB": ui["overheadGiB"],
            "note": {"es": ui["note"][0], "en": ui["note"][1]},
            "policyNote": {"es": ui["policyNote"][0], "en": ui["policyNote"][1]},
        })
    return out


# ---------------------------------------------------------------- platforms
APPLE_NOTE = ("macOS deja a la GPU 2/3 de la RAM con 32 GiB o menos y 3/4 por encima (se puede subir con "
              "sudo sysctl iogpu.wired_limit_mb).",
              "macOS lets the GPU use 2/3 of RAM at 32 GiB or less and 3/4 above (raise it with "
              "sudo sysctl iogpu.wired_limit_mb).")
CPU_RAM = {"cpu-ddr5-5600-2ch": [32, 64, 128, 256], "cpu-epyc9004-12ch": [256, 512, 1024]}
CPU_NOTE = ("Supuesto: se reservan 4 GB para el sistema operativo.", "Assumption: 4 GB are kept for the operating system.")


def price_of(p: dict, ram: int | None = None):
    op = p.get("official_price")
    if not op:
        return None
    if isinstance(op, list):
        match = [x for x in op if re.search(rf"\b{ram}\s*GB,", x.get("config", ""))] if ram else []
        if not match:
            return None
        op = match[0]
    label = {"apple-es-store": "Apple Store España", "nvidia-es-store": "NVIDIA España, PVPR",
             "launch-msrp": "precio de lanzamiento"}.get(op.get("kind"), op.get("kind") or "")
    config = re.split(r"\s*[;(]", op.get("config", ""))[0].strip()
    return {"amount": op["amount"], "currency": op["currency"], "label": f"{label} · {config}" if config else label}


def build_platforms(raw: dict) -> list[dict]:
    out = []
    for p in raw["platforms"]:
        # The "CPU" backend is offered only on CPU-only (x86) machines. Unified-memory
        # boxes list it too, but running there on the CPU is not the use case, and it
        # would let x86-CPU engine support (e.g. vLLM) match an ARM Mac.
        backends = p["backends"] if p["kind"] == "cpu" else [b for b in p["backends"] if b != "CPU"]
        base = {"vendor": p["vendor"], "bandwidth": p["memory_bandwidth_gbps"], "power": p["power_w"],
                "backends": backends}
        if p["kind"] == "discrete":
            out.append(dict(base, id=p["id"], name=p["name"], kind="discrete", memGiB=p["memory_gb"],
                            usableFraction=1, price=price_of(p)))
        elif p["vendor"] == "Apple":
            for ram in p["memory_options_gb"] or [p["memory_gb"]]:
                fr = (p["gpu_usable_memory"].get("fraction_by_ram_gb") or {}).get(str(ram))
                fr = fr if fr is not None else (2 / 3 if ram <= 32 else 0.75)
                out.append(dict(base, id=f"{p['id']}-{ram}", name=f"{p['name']} · {ram} GB", kind="unified",
                                memGiB=ram, usableFraction=round(fr, 4), price=price_of(p, ram),
                                usableNote={"es": APPLE_NOTE[0], "en": APPLE_NOTE[1]}))
        elif p["id"] == "strix-halo-395-128":
            g = p["gpu_usable_memory"]
            out.append(dict(base, id=p["id"] + "-win", name=p["name"] + " · Windows", kind="unified", memGiB=128,
                            usableFraction=g["fraction"], price=None,
                            usableNote={"es": "En Windows, AMD permite dedicar hasta 96 GB de 128 a la GPU.",
                                        "en": "On Windows, AMD lets you dedicate up to 96 GB of 128 to the GPU."}))
            out.append(dict(base, id=p["id"] + "-linux", name=p["name"] + " · Linux", kind="unified", memGiB=128,
                            usableFraction=g["linux_default_gtt_fraction"], price=None,
                            usableNote={"es": "En Linux la GPU usa por defecto ≈50 % de la RAM (GTT); se puede ampliar en el kernel.",
                                        "en": "On Linux the GPU uses ≈50% of RAM by default (GTT); the kernel limit can be raised."}))
        elif p["id"] == "dgx-spark":
            visible = min(p["gpu_usable_memory"]["os_visible_total_gib"])
            out.append(dict(base, id=p["id"], name=p["name"], kind="unified", memGiB=visible, usableFraction=1,
                            price=price_of(p),
                            usableNote={"es": f"Sin reparto fijo: el sistema ve ≈{visible} GiB de 128 y la GPU usa lo que quede libre.",
                                        "en": f"No fixed split: the OS sees ≈{visible} GiB of 128 and the GPU uses whatever is free."}))
        elif p["kind"] == "cpu":
            for ram in CPU_RAM[p["id"]]:
                out.append(dict(base, id=f"{p['id']}-{ram}", name=f"{p['name']} · {ram} GB", kind="cpu", memGiB=ram,
                                usableFraction=1, reserveGiB=4, price=None,
                                usableNote={"es": CPU_NOTE[0], "en": CPU_NOTE[1]}))
        else:
            raise SystemExit(f"unhandled platform {p['id']}")
    return out


# ------------------------------------------------------------------- models
def build_models(raw: list) -> list[dict]:
    out = []
    for m in raw:
        state_bytes = 4 if "float32" in (m.get("notes") or "").lower() and m["attention"] == "ssm-hybrid" else 2
        out.append({
            "id": m["id"], "name": m["name"], "family": m["family"], "params": m["params_b"],
            "active": m["active_params_b"], "moe": bool(m["moe"]), "attention": m["attention"],
            "kvElemFull": m["kv_elements_per_token_full"], "kvElemSwa": m["kv_elements_per_token_swa"] or 0,
            "swaWindow": m["swa_window"], "kvLayersFull": m["kv_layers_full"], "kvLayersSwa": m["kv_layers_swa"] or 0,
            "stateElems": m.get("fixed_state_elements") or 0, "stateBytes": state_bytes,
            "maxContext": m["max_context"],
        })
    for m in out:
        for k in ("params", "kvElemFull"):
            if not m[k]:
                raise SystemExit(f"model {m['id']} lacks {k}")
    return out


# ----------------------------------------------------------- decision models
# Decision models (docs/marketing/llm-calculator-data/decision-models.json) are LLM-shaped,
# so memory uses the same weights + KV formula. What differs is the RUNTIME: most of them
# need a decision head or a read-out that a plain chat engine does not run. Per engine:
#   "yes"    the engine runs the model as a decision model
#   "helper" stock engine plus the model's own helper script (shim)
#   "no"     not supported, or it loads the backbone but the answer is meaningless
#   None     not documented
# Read from each model's runtimes in decision-models.json (2026-11-12): llama.cpp decision
# support is /v1/systemone (PR #29818, 2026-10-02) and the clef arch (PR #29831, 2026-10-03).
_ALL_NO = {"llamacpp": "no", "ollama": "no", "lmstudio": "no", "vllm": "no", "sglang": "no",
           "mlx-lm": "no", "omlx": "no", "exllamav3": "no"}
DECISION_RUNTIMES = {
    "clef-27b": dict(_ALL_NO, llamacpp="yes", ollama="yes", sglang="yes", **{"mlx-lm": "helper"}),
    "clef-flash-9b": dict(_ALL_NO, llamacpp="yes", ollama="yes", sglang="yes", vllm="yes", **{"mlx-lm": "helper"}),
    "clef-omni-30b-a3b": dict(_ALL_NO, vllm="yes"),
    "openjev-27b": dict(_ALL_NO, llamacpp="yes", vllm="helper", lmstudio=None, sglang=None, omlx=None,
                        exllamav3=None, **{"mlx-lm": "helper"}),
    "openjev-flash-9b": dict(_ALL_NO, llamacpp="yes", vllm="helper", lmstudio=None, sglang=None, omlx=None,
                             exllamav3=None, **{"mlx-lm": "helper"}),
    "d3": dict(_ALL_NO, llamacpp="yes", ollama="yes", vllm="yes", sglang=None),
    "d3-flash": dict(_ALL_NO, llamacpp="yes", ollama="yes", vllm="yes", sglang=None),
    "d3-mini": dict(_ALL_NO, llamacpp="yes", ollama="yes", vllm="yes", sglang=None),
    "pplx-decider-v1.2": dict(_ALL_NO, llamacpp="yes", sglang=None),
    "pplx-decider-flash-v1.2": dict(_ALL_NO, llamacpp="yes", sglang=None),
    "kev-27b": dict(_ALL_NO, llamacpp=None, **{"mlx-lm": None}),
    "nimble-v4-max-27b": dict(_ALL_NO),
    "h2o-lightning-4b": dict(_ALL_NO, llamacpp="yes", vllm="helper", lmstudio=None, sglang=None, omlx=None,
                             exllamav3=None, **{"mlx-lm": None}),
    "liquid-d1-3b": dict(_ALL_NO, llamacpp="yes", vllm="yes"),
}
# Decision head parameters that the HF safetensors total leaves out (Clef ships it apart).
DECISION_HEAD_B = {"clef-27b": 0.13, "clef-flash-9b": 0.12}
# Published speed, translated. (hardware es/en, figure es/en, source). Single request unless stated.
DECISION_REFS = {
    "clef-27b": [
        ("Workers AI (SGLang)", "Workers AI (SGLang)", "152 ms con ≈800 tokens, 305 ms con ≈3.400, 1,6 s con ≈16.000",
         "152 ms at ≈800 tokens, 305 ms at ≈3,400, 1.6 s at ≈16,000", "https://blog.cloudflare.com/clef-faster-cheaper-multimodal"),
        ("Apple M5 Max, MLX 4 bits", "Apple M5 Max, MLX 4-bit", "1,4 s con 1.000 tokens, 26 s con 16.000",
         "1.4 s at 1,000 tokens, 26 s at 16,000", "https://huggingface.co/mlx-community/clef-flash-4bit"),
    ],
    "clef-flash-9b": [
        ("Workers AI", "Workers AI", "mediana 39 ms por petición", "median 39 ms per request",
         "https://developers.cloudflare.com/changelog/post/2026-10-01-clef-workers-ai/"),
        ("Apple M5 Max, MLX 4 bits", "Apple M5 Max, MLX 4-bit", "0,31 s con 1.000 tokens, 7 s con 16.000",
         "0.31 s at 1,000 tokens, 7 s at 16,000", "https://huggingface.co/mlx-community/clef-flash-4bit"),
    ],
    "clef-omni-30b-a3b": [
        ("Workers AI", "Workers AI", "texto ≈130 ms, imagen ≈150 ms, vídeo de 21 s ≈1,5 s",
         "text ≈130 ms, image ≈150 ms, 21-s video ≈1.5 s", "https://blog.cloudflare.com/clef-faster-cheaper-multimodal"),
    ],
    "openjev-27b": [
        ("1 × H100, FP8, vLLM", "1 × H100, FP8, vLLM", "227 ms con 1.100 tokens, 923 ms con 8.100, 1,76 s con 15.100",
         "227 ms at 1,100 tokens, 923 ms at 8,100, 1.76 s at 15,100", "https://huggingface.co/openjev/openjev"),
        ("Apple M5 Max, MLX 8 bits", "Apple M5 Max, MLX 8-bit", "≈0,3 s por decisión corta", "≈0.3 s per short decision",
         "https://huggingface.co/openjev/openjev-MLX"),
    ],
    "openjev-flash-9b": [
        ("1 × H100, FP8", "1 × H100, FP8", "40 ms por decisión", "40 ms per decision", "https://huggingface.co/openjev/OpenJev-Flash-9B"),
    ],
    "d3": [
        ("1 × AMD MI325X", "1 × AMD MI325X", "54 ms una pregunta; 4,2 s con 32.000 tokens; 27 peticiones/s en lote",
         "54 ms one question; 4.2 s at 32,000 tokens; 27 requests/s batched", "https://huggingface.co/vllm-sr/d3"),
    ],
    "d3-flash": [
        ("1 × AMD MI325X", "1 × AMD MI325X", "19 ms una pregunta; 1,4 s con 32.000 tokens; 95 peticiones/s en lote",
         "19 ms one question; 1.4 s at 32,000 tokens; 95 requests/s batched", "https://huggingface.co/vllm-sr/d3-flash"),
    ],
    "d3-mini": [
        ("1 × AMD MI325X", "1 × AMD MI325X", "18 ms una pregunta; 1,1 s con 32.000 tokens; 151 peticiones/s en lote",
         "18 ms one question; 1.1 s at 32,000 tokens; 151 requests/s batched", "https://huggingface.co/vllm-sr/d3-mini"),
    ],
    "pplx-decider-v1.2": [
        ("1 × NVIDIA GB200", "1 × NVIDIA GB200", "96 ms una petición pequeña; 19,8 peticiones/s de ≈1.050 tokens",
         "96 ms one small request; 19.8 requests/s at ≈1,050 tokens", "https://huggingface.co/perplexity-ai/pplx-decider-v1.2"),
    ],
    "kev-27b": [
        ("1 × H200", "1 × H200", "9,4 s con 64.000 tokens nuevos (733 ms en caché); 21 peticiones/s",
         "9.4 s at 64,000 new tokens (733 ms cached); 21 requests/s", "https://huggingface.co/jaredpalmer/kev-27b"),
    ],
    "h2o-lightning-4b": [
        ("1 × H100", "1 × H100", "21 ms una pregunta (p50); 43 peticiones/s", "21 ms one question (p50); 43 requests/s",
         "https://huggingface.co/h2oai/h2o-lightning-4b"),
        ("1 × L40S", "1 × L40S", "28 ms con 100-300 tokens, 69 ms con 800-1.200", "28 ms at 100-300 tokens, 69 ms at 800-1,200",
         "https://huggingface.co/h2oai/h2o-lightning-4b"),
    ],
    "liquid-d1-3b": [
        ("RTX 4090", "RTX 4090", "8 ms una pregunta; 102 ms con 3.400 tokens", "8 ms one question; 102 ms at 3,400 tokens",
         "https://huggingface.co/LiquidAI/d1-3B"),
        ("Apple M5 Pro (PyTorch mps)", "Apple M5 Pro (PyTorch mps)", "30 ms una pregunta; 640 ms con 3.400 tokens",
         "30 ms one question; 640 ms at 3,400 tokens", "https://huggingface.co/LiquidAI/d1-3B"),
    ],
}


def build_decision_models(raw: dict) -> list[dict]:
    out = build_models(raw["models"])
    by_id = {m["id"]: m for m in raw["models"]}
    for m in out:
        src = by_id[m["id"]]
        if m["id"] not in DECISION_RUNTIMES:
            raise SystemExit(f"decision model {m['id']} has no DECISION_RUNTIMES row")
        m["params"] = round(m["params"] + DECISION_HEAD_B.get(m["id"], 0), 2)
        if not m["moe"]:
            m["active"] = m["params"]
        # LFM Open License v1.0 allows commercial use below USD 10M revenue: conditional, not "no".
        commercial = None if m["id"] == "liquid-d1-3b" else src["decision"].get("license_commercial")
        m["decision"] = {
            "runtimes": DECISION_RUNTIMES[m["id"]],
            "license": src["license"], "commercial": commercial if isinstance(commercial, bool) else None,
            "refs": [{"hw": {"es": r[0], "en": r[1]}, "fig": {"es": r[2], "en": r[3]}, "source": r[4]}
                     for r in DECISION_REFS.get(m["id"], [])],
        }
    return out


def build_calibration(raw: dict) -> dict:
    subs = raw["calibration"]["k_subsets"]["gpu_all_backends_by_ceiling_band"]
    bands = []
    for key, v in sorted(subs.items(), key=lambda kv: float(kv[0].split("-")[0])):
        hi = key.split("-")[1]
        bands.append({"max": None if hi == "inf" else float(hi), "k": [v["p25"], v["p75"]], "n": v["n"]})
    cpu = raw["calibration"]["k_by_backend"]["CPU"]
    # Highest ceiling any measurement reached (same formula as hardware.json's
    # calibration.formula): beyond it the tool has no evidence speed keeps scaling.
    ceilings = [p["bandwidth_gbps"] / (p["size_gib"] * 1.073741824) for p in raw["calibration"]["points"]
                if p.get("bandwidth_gbps") and p.get("size_gib")]
    cal = {"cpu": [cpu["p25"], cpu["p75"]], "bands": bands, "maxCeiling": round(max(ceilings), 1),
           "n": len(ceilings)}
    moe_file = DATA / "moe-calibration.json"
    if moe_file.exists():
        cal["moe"] = build_moe_calibration(json.loads(moe_file.read_text()))
    return cal


def build_moe_calibration(raw: dict) -> dict:
    """MoE decode efficiency, measured on MoE models (all calibration points above are dense)."""
    k = raw["k_moe"]
    edges = {"<100": 100.0, "100-250": 250.0, "250-500": 500.0, ">=500": None}
    bands = [{"max": edges[key], "k": [k["by_ceiling_band"][key]["p25"], k["by_ceiling_band"][key]["p75"]],
              "n": k["by_ceiling_band"][key]["n"]} for key in ("<100", "100-250", "250-500", ">=500")]
    cpu = k["by_backend"]["CPU"]
    return {"cpu": [cpu["p25"], cpu["p75"]], "bands": bands, "n": k["all"]["n"],
            "maxCeiling": k.get("max_ceiling_active")}


# ------------------------------------------------------------------ Colibri
# Expert streaming does not fit the weights + KV model, so it carries the project's
# own requirement table (README v1.11.0) instead. Texts translate the README rows.
COLI_GPU = {
    "not needed": ("No necesita GPU.", "No GPU needed."),
    "optional NVIDIA": ("GPU NVIDIA opcional: el proyecto mide el prefill 5-10 veces y la generación ≈2,5 veces más rápidos.",
                        "Optional NVIDIA GPU: the project measures prefill 5-10x and decode ≈2.5x faster."),
    "not supported (CPU only)": ("Solo CPU: no admite GPU.", "CPU only: no GPU support."),
    "optional CUDA VRAM expert tier": ("Con CUDA puede tener expertos en VRAM (1,44 → 10,05 tok/s en dos tarjetas de 8 GB, según el proyecto).",
                                       "With CUDA it can keep experts in VRAM (1.44 → 10.05 tok/s on two 8 GB cards, per the project)."),
}
COLI_HW = {
    "6x RTX 5090, full residency": ("6 × RTX 5090, todo residente", "6 × RTX 5090, fully resident"),
    "128 GB CPU-only desktop, warm": ("sobremesa de 128 GB solo CPU, caché caliente", "128 GB CPU-only desktop, warm cache"),
    "single RTX 5070 Ti, GPU-resident pipeline": ("una RTX 5070 Ti, tubería residente en GPU", "one RTX 5070 Ti, GPU-resident pipeline"),
    "25 GB dev box, cold": ("equipo de 25 GB, en frío", "25 GB machine, cold"),
    "CPU vs CUDA VRAM expert tier on two 8 GB cards": ("CPU frente a expertos en VRAM con dos tarjetas de 8 GB", "CPU vs experts in VRAM on two 8 GB cards"),
    "linux/arm64 VM, 18 cores, no GPU, shared load": ("VM linux/arm64 de 18 núcleos sin GPU, con carga compartida", "linux/arm64 VM, 18 cores, no GPU, shared load"),
}
COLI_BY = {"project": ("proyecto", "project"), "community": ("comunidad", "community"), "jacar.es": ("medido por jacar.es", "measured by jacar.es")}


def build_colibri(raw: dict) -> tuple[dict, dict]:
    e = raw["engine"]
    models = [{
        "id": m["id"], "name": m["name"], "totalB": m["total_b"], "activeB": m["active_b"], "container": m["container"],
        "diskGB": m["disk_gb"], "ramMinGB": m["ram_min_gb"], "ramComfortableGB": m["ram_comfortable_gb"],
        "expertGBPerToken": m["expert_gb_per_token"],
        "gpu": ({"es": COLI_GPU[m["gpu"]][0], "en": COLI_GPU[m["gpu"]][1]} if m["gpu"] else None),
    } for m in raw["models"]]
    refs = [{"model": r["model"], "tokps": r["tokps"], "source": r["source"],
             "hardware": {"es": COLI_HW[r["hardware"]][0], "en": COLI_HW[r["hardware"]][1]},
             "by": {"es": COLI_BY[r["by"]][0], "en": COLI_BY[r["by"]][1]}} for r in raw["references"]]
    colibri = {"version": e["version"], "source": e["source"], "models": models, "references": refs}
    engine = {
        "id": "colibri", "name": "Colibri", "version": e["version"], "kind": "expert-streaming",
        # Every Colibri family runs on the CPU, so it is compatible with any platform;
        # CUDA, Metal and Vulkan only accelerate.
        "backends": ["CPU", "CUDA", "Metal", "Vulkan", "ROCm", "SYCL"],
        "formats": [{"id": "container", "label": "Contenedor del proyecto", "bpw": 4}], "defaultFormat": "container",
        "kvDtypes": [{"id": "native", "label": "Nativa", "bytes": 1}], "policy": {"type": "on-demand"},
        "swaWindowOnly": None, "swaPadTokens": 0, "mla": None, "linearHybrid": None, "ssmHybrid": None, "overheadGiB": 0,
        "note": {"es": "Deja en RAM la parte densa y lee del SSD los expertos que elige el router. Aquí no se compara con la VRAM: "
                       "cuenta la RAM del sistema frente al mínimo que publica el proyecto y el disco libre frente al tamaño del "
                       "modelo. La velocidad la marca el disco.",
                 "en": "Keeps the dense part in RAM and reads the experts the router picks from the SSD. It is not compared with "
                       "VRAM here: system RAM is checked against the project's published minimum and free disk against the model "
                       "size. The disk sets the speed."},
        "policyNote": {"es": f"Requisitos por modelo tomados de la tabla del README de Colibri {e['version']}.",
                       "en": f"Per-model requirements from the Colibri {e['version']} README table."},
    }
    return colibri, engine


# ---------------------------------------------------------------- voice (TTS)
# Text-to-speech does not fit the weights + KV + decode model (codec models, vocoders,
# flow matching), so it gets its own panel built from data/
# tts-models.json (2026-11-12). The facts come from that file; what follows is editorial:
#   TTS_SPANISH  how Spanish is supported: yes (official) | variant (only a separate
#                checkpoint or finetune) | experimental | no, with a short bilingual note
#   TTS_MEM      the memory figures the authors or the community PUBLISH, as numbers, per
#                platform kind; the browser picks the smallest one that applies. A model
#                without one is estimated as its 16-bit weights and labelled as a minimum.
#   TTS_COMMERCIAL_NOTE  licences whose commercial use depends on conditions
TTS_SPANISH = {
    "kokoro-82m": ("yes", "3 voces oficiales; el autor avisa de que fuera del inglés la calidad puede ser escasa.",
                   "3 official voices; the author warns that quality outside English may be thin."),
    "piper": ("yes", "Voces nativas de España (davefx, sharvard), México y Argentina.", "Native voices from Spain (davefx, sharvard), Mexico and Argentina."),
    "melotts": ("yes", "Modelo español oficial de una sola voz.", "Official single-voice Spanish model."),
    "supertonic-3": ("yes", "Oficial; WER en español 1,13 según el proyecto.", "Official; Spanish WER 1.13 per the project."),
    "kitten-tts-2": ("yes", "Voces en español integradas o clonación; desactiva la normalización inglesa (normalize=False).",
                     "Built-in Spanish voices or cloning; turn off the English normaliser (normalize=False)."),
    "parler-tts-mini-multilingual-v1.1": ("yes", "Oficial; entrenado con audiolibros en español.", "Official; trained on Spanish audiobooks."),
    "omnivoice": ("yes", "Oficial (27.500 h de español); el acento sigue a la voz de referencia.", "Official (27,500 h of Spanish); the accent follows the reference voice."),
    "neutts-air": ("no", "Solo inglés.", "English only."),
    "neutts-nano": ("variant", "Solo con el modelo aparte neutts-nano-spanish.", "Only through the separate neutts-nano-spanish model."),
    "moss-tts-nano-100m": ("yes", "Oficial; sin cifras de calidad en español.", "Official; no Spanish quality figures."),
    "orpheus-3b": ("variant", "Solo con el modelo beta de investigación español-italiano.", "Only through the Spanish-Italian research beta checkpoint."),
    "sesame-csm-1b": ("no", "Solo inglés.", "English only."),
    "dia-1.6b": ("no", "Solo inglés.", "English only."),
    "dia2-2b": ("no", "Solo inglés.", "English only."),
    "spark-tts-0.5b": ("no", "Solo chino e inglés.", "Chinese and English only."),
    "llasa-3b": ("no", "Solo chino e inglés (el español está en otro modelo, Llasa-1B-Multilingual).",
                 "Chinese and English only (Spanish is in a different model, Llasa-1B-Multilingual)."),
    "zonos-v0.1": ("experimental", "No oficial: hubo español en el entrenamiento, pero no está en la lista de idiomas.",
                   "Unofficial: Spanish was in the training data but is not in the language list."),
    "zonos3-mini": ("yes", "Oficial, con normalización de números y fechas; voces Rodrigo y Ximena.",
                    "Official, with number and date normalisation; voices Rodrigo and Ximena."),
    "kyutai-tts-1.6b-en-fr": ("no", "Solo inglés y francés.", "English and French only."),
    "xtts-v2": ("yes", "Oficial; el acento sigue al clip de referencia.", "Official; the accent follows the reference clip."),
    "chatterbox-multilingual-v3": ("yes", "Oficial, y además modelos afinados para español de España (es-ES) y de México.",
                                   "Official, plus finetunes for Spain Spanish (es-ES) and Mexican Spanish."),
    "f5-tts-v1-base": ("variant", "Solo con el ajuste fino de la comunidad F5-Spanish (sobre todo acento peninsular).",
                       "Only through the community F5-Spanish finetune (mostly Peninsular accent)."),
    "openaudio-s1-mini": ("yes", "Listado como soportado, sin datos de calidad.", "Listed as supported, no quality data."),
    "fish-audio-s2-pro": ("yes", "Nivel 2 de calidad (el 1 es japonés, inglés y chino).", "Quality tier 2 (tier 1 is Japanese, English and Chinese)."),
    "fish-audio-drama-3": ("yes", "Oficial; WER en español 0,96 según el informe técnico.", "Official; Spanish WER 0.96 per the technical report."),
    "indextts-2": ("no", "Sin español (llegó con la 2.5).", "No Spanish (it arrived in 2.5)."),
    "indextts-2-5": ("yes", "Oficial desde la 2.5; WER en español 3,75.", "Official since 2.5; Spanish WER 3.75."),
    "fun-cosyvoice3-0-5b": ("yes", "Uno de sus 9 idiomas oficiales; WER en español 4,04.", "One of its 9 official languages; Spanish WER 4.04."),
    "megatts3": ("no", "Solo chino e inglés.", "Chinese and English only."),
    "qwen3-tts-12hz-1.7b": ("yes", "Oficial, pero ninguna voz predefinida es nativa: clona o diseña una voz española.",
                            "Official, but no preset voice is native: clone or design a Spanish voice."),
    "qwen3-tts-12hz-0.6b": ("yes", "Oficial; menos preciso que el 1.7B y sin voz nativa predefinida.", "Official; less accurate than the 1.7B and no native preset voice."),
    "voxtral-4b-tts-2603": ("yes", "Voces predefinidas es_male y es_female; WER independiente 3,91, peor que Qwen3-TTS o VoxCPM2.",
                            "Preset voices es_male and es_female; independent WER 3.91, behind Qwen3-TTS or VoxCPM2."),
    "voxcpm2": ("yes", "Oficial; WER en español 2,0 en el Open TTS Leaderboard.", "Official; Spanish WER 2.0 on the Open TTS Leaderboard."),
    "vibevoice-realtime-0.5b": ("experimental", "Experimental; WER en español 7,37 en el Open TTS Leaderboard.", "Experimental; Spanish WER 7.37 on the Open TTS Leaderboard."),
    "vibevoice-1.5b": ("no", "Solo inglés y chino.", "English and Chinese only."),
    "higgs-tts-3-4b": ("yes", "Calidad de producción con voz de referencia: WER 1,5 clonando y 10,2 sin ella.",
                       "Production quality with a reference voice: WER 1.5 when cloning, 10.2 without."),
    "higgs-tts-2-3b-base": ("experimental", "No oficial (solo lo anuncia mlx-audio).", "Unofficial (only mlx-audio lists it)."),
    "moss-tts-v1.5": ("yes", "Oficial; indica language=\"Spanish\".", "Official; set language=\"Spanish\"."),
    "magpie-tts-multilingual-357m": ("yes", "Oficial, pero las voces predefinidas son inglesas.", "Official, but the preset voices are English."),
}
# (GB, platform kinds or None for any, condition es, condition en). GiB figures are kept as published.
TTS_MEM = {
    "supertonic-3": [(2.0, None, "pico en CPU", "peak on CPU")],
    "kitten-tts-2": [(5.5, ["discrete"], "pico en una GTX 1660 Ti, float16 + 4 bits", "peak on a GTX 1660 Ti, float16 + 4-bit"),
                     (6.0, ["cpu", "unified"], "≈6 GB de RAM en CPU, según el proyecto", "≈6 GB of RAM on CPU, per the project")],
    "omnivoice": [(5.0, ["discrete"], "pico en una A100 (medido por Supertone)", "peak on an A100 (measured by Supertone)"),
                  (8.3, ["cpu", "unified"], "pico en CPU (medido por Supertone)", "peak on CPU (measured by Supertone)")],
    "dia-1.6b": [(4.4, None, "bfloat16 en una RTX 4090", "bfloat16 on an RTX 4090")],
    "dia2-2b": [(3.9, ["discrete"], "GGUF Q4_0 con audio.cpp, RX 9070 (comunidad)", "GGUF Q4_0 with audio.cpp, RX 9070 (community)")],
    "zonos-v0.1": [(6.0, None, "requisito del proyecto", "project requirement")],
    "fish-audio-s2-pro": [(4.0, None, "GGUF Q4_K_M con s2.cpp (guía de la comunidad)", "GGUF Q4_K_M with s2.cpp (community guide)"),
                          (24.0, None, "PyTorch, documentación oficial", "PyTorch, official docs")],
    "fish-audio-drama-3": [(26.0, ["unified"], "MLX 4 bits en Apple Silicon, pico (comunidad)", "MLX 4-bit on Apple Silicon, peak (community)"),
                           (48.0, ["discrete"], "FP8, documentación oficial", "FP8, official docs")],
    "indextts-2-5": [(6.0, ["discrete"], "GPU NVIDIA, según la ficha oficial", "NVIDIA GPU, per the official card")],
    "voxtral-4b-tts-2603": [(16.0, ["discrete", "unified"], "BF16 en una GPU con vLLM-Omni", "BF16 on one GPU with vLLM-Omni"),
                            (8.0, ["cpu", "unified"], "GGUF Q4_K en CPU: cabe en 8 GB de RAM", "GGUF Q4_K on CPU: fits in 8 GB of RAM")],
    "voxcpm2": [(8.0, None, "bf16 con PyTorch, ficha oficial", "bf16 with PyTorch, official card")],
    "higgs-tts-3-4b": [(6.0, ["discrete"], "GGUF Q8 con audio.cpp, pico", "GGUF Q8 with audio.cpp, peak"),
                       (8.6, ["unified"], "bf16 en un Apple M5 Pro, pico", "bf16 on an Apple M5 Pro, peak")],
    "moss-tts-v1.5": [(7.3, ["discrete"], "GGUF Q4_K con audio.cpp, RTX 3090 (comunidad)", "GGUF Q4_K with audio.cpp, RTX 3090 (community)")],
}
TTS_COMMERCIAL_NOTE = {
    "piper": ("Motor GPL-3; cada voz lleva la licencia de sus datos: davefx, sharvard y carlfm (es_ES) son aptas para uso comercial.",
              "GPL-3 engine; each voice carries its dataset licence: davefx, sharvard and carlfm (es_ES) are cleared for commercial use."),
    "supertonic-3": ("OpenRAIL-M: comercial con restricciones de uso.", "OpenRAIL-M: commercial with use restrictions."),
    "kitten-tts-2": ("Gratis por debajo de 1 M USD de ingresos o financiación.", "Free below USD 1M of revenue or funding."),
    "neutts-nano": ("Licencia de acceso restringido: no se ha podido leer si permite uso comercial.", "Gated licence: whether it allows commercial use could not be read."),
    "fish-audio-drama-3": ("Por debajo de 500.000 USD de ingresos al año y sin ofrecer una API de TTS.", "Below USD 500k of yearly revenue and without offering a TTS API."),
    "indextts-2": ("Licencia aparte por encima de 100 M de usuarios al mes.", "Separate licence above 100M monthly users."),
    "indextts-2-5": ("Licencia aparte por encima de 100 M de usuarios al mes.", "Separate licence above 100M monthly users."),
    "vibevoice-realtime-0.5b": ("MIT, pero Microsoft dice que es solo para investigación.", "MIT, but Microsoft says it is for research only."),
    "vibevoice-1.5b": ("MIT, pero Microsoft dice que es solo para investigación.", "MIT, but Microsoft says it is for research only."),
    "higgs-tts-3-4b": ("No comercial, salvo contenido monetizado de creadores con atribución.", "Non-commercial, except monetised creator content with attribution."),
    "higgs-tts-2-3b-base": ("Comercial por debajo de 100.000 usuarios activos al año.", "Commercial below 100k yearly active users."),
}
# Commercial use that depends on conditions: neither a clean yes nor a clean no.
TTS_CONDITIONAL = {"piper", "supertonic-3", "kitten-tts-2", "fish-audio-drama-3", "vibevoice-realtime-0.5b",
                   "vibevoice-1.5b", "higgs-tts-2-3b-base"}
TTS_BACKEND = {"CUDA": "CUDA", "ROCm": "ROCm", "Metal/MPS": "Metal", "CPU": "CPU", "other: Vulkan": "Vulkan",
               "WebGPU": "WebGPU", "WASM": "WASM"}
TTS_BY = {"project": ("proyecto", "project"), "community": ("comunidad", "community"), "paper": ("artículo", "paper")}


def build_tts(raw: dict) -> list[dict]:
    out = []
    for m in raw["models"]:
        mid = m["id"]
        if mid not in TTS_SPANISH:
            raise SystemExit(f"TTS model {mid} has no TTS_SPANISH row")
        sp = TTS_SPANISH[mid]
        backends = sorted({TTS_BACKEND[b] for r in m["runtimes"] for b in (r.get("backends") or []) if b in TTS_BACKEND})
        params = m["params_b"]
        if params:
            est = params * 1e9 * 2 / 1073741824
        else:  # Piper: no parameter count published; the largest Spanish voice file stands in
            est = max(w["gb"] for w in m["weights_gb"] if w.get("gb"))
        com = m["license_commercial"]
        if mid in TTS_CONDITIONAL:
            com = "conditions"
        cn = TTS_COMMERCIAL_NOTE.get(mid)
        speeds = [{"hw": None if re.search(r"not named|unspecified|not stated", s["hardware"] or "not named", re.I) else s["hardware"], "runtime": s.get("runtime"), "metric": s["metric"], "value": s["value"],
                   "platform": s.get("platform_id"), "by": {"es": TTS_BY.get(s.get("by"), (s.get("by"), s.get("by")))[0],
                                                             "en": TTS_BY.get(s.get("by"), (s.get("by"), s.get("by")))[1]},
                   "source": s["source"]}
                  for s in (m.get("speed_published") or []) if isinstance(s.get("value"), (int, float))]
        out.append({
            # The research names carry English asides in parentheses ("(+ community Spanish
            # finetune)"); the Spanish note says the same in both languages, so drop them.
            "id": mid, "name": m["name"].split(" (")[0], "vendor": m["vendor"], "repo": m["repo"], "params": params,
            "released": m["released"], "license": m["license"].split(";")[0].split(" (")[0],
            "commercial": com, "commercialNote": {"es": cn[0], "en": cn[1]} if cn else None,
            "spanish": sp[0], "spanishNote": {"es": sp[1], "en": sp[2]},
            "cloning": bool(m["voice_cloning"]["supported"]), "streaming": m["streaming"]["supported"],
            "sampleRate": m.get("sample_rate_hz"),
            "runtimes": sorted({r["name"].split(" (")[0] for r in m["runtimes"] if r["name"] != "other"}),
            "backends": backends,
            "mem": [{"gb": g, "kinds": k, "cond": {"es": ce, "en": cen}} for g, k, ce, cen in TTS_MEM.get(mid, [])],
            "estGB": round(est, 2), "speeds": speeds,
        })
    return out


def render() -> str:
    models = build_models(json.loads((DATA / "models.json").read_text()))
    models += build_decision_models(json.loads((DATA / "decision-models.json").read_text()))
    hardware = json.loads((DATA / "hardware.json").read_text())
    engines = build_engines(json.loads((DATA / "engines.json").read_text()))
    colibri, coli_engine = build_colibri(json.loads((DATA / "colibri.json").read_text()))
    engines.append(coli_engine)
    dump = lambda o: json.dumps(o, ensure_ascii=False, separators=(",", ":"))
    return ("/* DATA:BEGIN */\n"
            "  // GENERATED by scripts/build-llm-calc-data.py from data/. Do not edit.\n"
            f"  var DATA_VERIFIED = {dump(VERIFIED)};\n"
            f"  var MODELS = {dump(models)};\n"
            f"  var PLATFORMS = {dump(build_platforms(hardware))};\n"
            f"  var ENGINES = {dump(engines)};\n"
            f"  var CALIBRATION = {dump(build_calibration(hardware))};\n"
            f"  var COLIBRI = {dump(colibri)};\n"
            f"  var TTS = {dump(build_tts(json.loads((DATA / 'tts-models.json').read_text())))};\n"
            "  /* DATA:END */")


def main() -> int:
    src = JS.read_text(encoding="utf-8")
    block = render()
    new = re.sub(r"/\* DATA:BEGIN \*/.*?/\* DATA:END \*/", lambda _: block, src, count=1, flags=re.S)
    if "--check" in sys.argv:
        return 0 if new == src else 1
    JS.write_text(new, encoding="utf-8")
    print(f"wrote data block ({len(block):,} chars) into {JS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
