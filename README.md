# Local LLM calculator data

The sourced dataset and the generator behind the **local LLM calculator** on jacar.es:

- Spanish: <https://jacar.es/calculadora-llm-local/>
- English: <https://jacar.es/en/local-llm-calculator/>

The calculator answers three questions before you buy hardware or download weights:
does a model fit in memory with a given engine, context and quantization; roughly
how fast will it decode; and what does running it yourself cost compared with an API.
It also covers decision models (classifiers that return a probability per option)
and open text-to-speech models, each in its own section.
This repository holds the data and the pure compute functions. The page markup
is not included.

## What is in it

| Path | Contents |
|---|---|
| `data/models.json` | 60 open-weight chat models: parameters (total and active), attention type (full, sliding-window hybrid, MLA, linear/SSM hybrids), KV elements per token split into full and sliding-window layers, window size, fixed state, max context |
| `data/hardware.json` | 34 platforms (discrete GPUs, Apple Silicon, AMD Strix Halo, NVIDIA DGX Spark, CPU-only servers): memory, bandwidth, power, how much memory the GPU may use, official prices where the vendor publishes one; plus 132 dense-model `llama-bench` points used to calibrate the speed estimate |
| `data/moe-calibration.json` | Mixture-of-experts speed calibration points and the derived efficiency factors |
| `data/engines.json` | 8 inference engines, each pinned to a release (llama.cpp, Ollama, LM Studio, vLLM, SGLang, MLX-LM, oMLX, ExLlamaV3): weight formats with bits per weight, KV cache types with bytes per element, memory reservation policy, attention support |
| `data/decision-models.json` | 14 decision models (Clef, OpenJev, d3, pplx-decider and others): the same architecture fields as `models.json`, plus the base model, how the answer is read out, which engines run it, published throughput and licence |
| `data/tts-models.json` | 39 open text-to-speech models: components, languages, voice cloning, streaming, licence, weights per format, runtimes and backends, published memory and published speed, each with its source |
| `data/colibri.json` | Colibri expert-streaming engine: per-model RAM and disk requirements and published speed references |
| `data/*-notes.md` | Research notes: how each figure was derived, where sources disagree and which one was used |
| `scripts/build-llm-calc-data.py` | Generator: reads `data/` and writes the data block of `calculator/llm-calculator.js` |
| `calculator/llm-calculator.js` | The calculator: memory, KV cache, speed and cost functions (UMD; works in node and the browser) |
| `tests/test_llm_calculator.cjs` | Hand-computed fixtures plus invariants on the shipped data |

## How the data is sourced

Every value carries the source it was read from. Model entries list the
Hugging Face `config.json` and model API URLs they come from; platforms list the
vendor spec pages (and the formula when bandwidth is computed from bus width and
memory clock); engines are pinned to a release tag and commit, with the file in
the engine's own source where each behaviour was read; calibration points link
the benchmark run they were taken from. Where the documentation and the code
disagree, the code wins and the notes say so (for example, vLLM's
`gpu_memory_utilization` default is 0.92 in the v0.29.0 code while the docs say
0.9). Undocumented behaviour is costed as the expensive case and labelled.

Prices appear only where the vendor publishes them. Platforms and engines were
verified on 2026-09-15; chat, decision and voice models on 2026-11-12
(`DATA_VERIFIED` in the generated block).

On top of the verified facts, the generator adds a few editorial choices, all
visible in `scripts/build-llm-calc-data.py`: which formats and KV types each engine
offers, the bilingual notes, the RAM variants a platform expands into, and three
labelled assumptions (4 GiB kept for the OS on CPU-only machines, each engine's
flat overhead, SGLang's Metal runtime left out).

### Speed model in one paragraph

Decode is memory-bandwidth bound: the ceiling is bandwidth divided by the bytes
read per token (active weights plus one sequence's KV cache). Real engines reach
a fraction `k` of that ceiling. `k` is calibrated from measured `llama-bench`
results: by ceiling band for dense models (132 points) and separately for MoE
models, which reach a lower fraction. The estimate is capped at the highest
ceiling seen in the calibration set, because there is no evidence that speed keeps
scaling beyond it.

### Decision models

A decision model is an LLM with a head that scores a fixed set of options in one
forward pass, so its memory is the same weights + KV formula as a chat model (plus
the separately shipped head where there is one). Speed is not estimated: one pass
over the prompt is compute bound, and the calibration above only covers
bandwidth-bound decode. The calculator quotes the published throughput figures
instead. Support differs by engine: `yes`, `helper` (needs the model's own helper
script), `no` (the engine loads the weights but returns a meaningless answer) or
undocumented.

### Voice models

Text-to-speech models (codec language models, vocoders, flow matching) do not fit
the weights + KV + decode model, so they have their own section. Memory is the
smallest figure the authors or the community publish for that kind of platform
(discrete GPU, Apple Silicon, CPU); when nothing is published, the 16-bit weights
stand in and the result is labelled as a minimum. Speeds are only the published
ones (real-time factor, times real time, time to first audio) with the hardware
they were measured on. Spanish support (official, separate checkpoint only,
experimental, none) and commercial-use conditions are editorial summaries in the
generator.

## Regenerate and test

Requirements: Python 3.9+ and Node.js 18+. No dependencies to install.

```sh
make build   # rewrite the data block of calculator/llm-calculator.js from data/
make test    # fail if the block is stale, then run the unit tests
```

Without `make`:

```sh
python3 scripts/build-llm-calc-data.py           # regenerate
python3 scripts/build-llm-calc-data.py --check   # exit 1 if the block is stale
node tests/test_llm_calculator.cjs               # unit tests (prints ALL PASS)
```

Edit the JSON, never the generated block between `/* DATA:BEGIN */` and
`/* DATA:END */`.

## Corrections

If a value is wrong or a source has changed, open an issue with the URL that
shows the correct figure. Changes without a source are not accepted, because the
point of the dataset is that every number can be checked.

## Licence

- Code (`scripts/`, `calculator/`, `tests/`): [MIT](LICENSE).
- Data (`data/`): [CC BY 4.0](LICENSE-data). Credit "Javier Cañete (jacar.es)" and
  link to <https://jacar.es/en/local-llm-calculator/>.
