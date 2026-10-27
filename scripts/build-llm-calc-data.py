#!/usr/bin/env python3
"""Generate the data block of the local LLM calculator from verified JSON.

Sources (verified against primary sources on 2026-09-15, every value with its URL):
  data/models.json           model architectures (config.json, HF API)
  data/hardware.json         platforms + llama.cpp speed calibration (dense models)
  data/moe-calibration.json  speed calibration on MoE models
  data/engines.json          engines pinned to a release tag
  data/colibri.json          Colibri expert-streaming catalogue

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
VERIFIED = "2026-09-15"


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


def render() -> str:
    models = build_models(json.loads((DATA / "models.json").read_text()))
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
