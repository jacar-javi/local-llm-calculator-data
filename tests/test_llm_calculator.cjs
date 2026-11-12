/* Unit tests for the LLM-calculator pure formulas (calculator/llm-calculator.js).
 * Run: node tests/test_llm_calculator.cjs
 *
 * Part A pins the arithmetic with fixture models whose expected values are
 * computed by hand in the comments. Part B checks the shipped data against
 * facts verified on 2026-09-15 (config.json, vendor specs, engine source), so a
 * data edit that breaks an architectural invariant fails loudly.
 * Memory is in GiB (2^30 bytes), the unit GPU and Mac memory is sold in. */
const path = require('path');
const calc = require(path.join(__dirname, '..', 'calculator', 'llm-calculator.js'));

let failures = 0;
function approx(name, got, want, tol) {
  tol = tol === undefined ? 1e-6 : tol;
  const ok = typeof got === 'number' && Math.abs(got - want) <= tol;
  if (!ok) { failures++; console.error(`FAIL ${name}: got ${got}, want ${want} (±${tol})`); }
  else { console.log(`ok   ${name}: ${got}`); }
}
function assert(name, cond) {
  if (!cond) { failures++; console.error(`FAIL ${name}`); } else { console.log(`ok   ${name}`); }
}
const GIB = 1073741824;

// ---------------------------------------------------------------- A. fixtures
// KV elements per token, K and V counted: dense Llama-3-8B shape,
// 2 x 32 layers x 8 KV heads x head_dim 128 = 65,536.
const FULL = { id: 'f', params: 8, active: 8, kvElemFull: 65536, kvElemSwa: 0, swaWindow: null,
               stateElems: 0, stateBytes: 2, attention: 'full', maxContext: 131072 };
// Sliding-window hybrid: 10 global layers (2 x 10 x 16 x 128 = 40,960) and
// 50 local layers (2 x 50 x 16 x 128 = 409,600) with a 1,024-token window.
const SWA = { id: 's', params: 30, active: 30, kvElemFull: 40960, kvElemSwa: 409600, swaWindow: 1024,
              stateElems: 0, stateBytes: 2, attention: 'swa-hybrid', maxContext: 262144 };
// MLA, DeepSeek-V3 shape: one latent per layer, 61 x (512 + 64) = 35,136 (no K/V pair).
const MLA = { id: 'm', params: 671, active: 37, kvElemFull: 35136, kvElemSwa: 0, swaWindow: null,
              stateElems: 0, stateBytes: 2, attention: 'mla', maxContext: 163840 };

const ENG_LOCAL = { id: 'local', backends: ['CUDA', 'Metal', 'CPU'], swaWindowOnly: true, swaPadTokens: 0, mla: true,
                    linearHybrid: true, ssmHybrid: true, overheadGiB: 1, policy: { type: 'ctx-prealloc' } };
const ENG_NOSWA = Object.assign({}, ENG_LOCAL, { id: 'noswa', swaWindowOnly: false });
const ENG_VLLM = Object.assign({}, ENG_LOCAL, { id: 'vllm', backends: ['CUDA'], mla: false,
                    policy: { type: 'fraction', fraction: 0.92 } });

// Full attention: 65,536 elements x 2 B = 131,072 B/token; x 8,192 = 2^30 B = 1 GiB.
let kv = calc.kvCache(FULL, { kvBytes: 2, context: 8192, parallel: 1, engine: ENG_LOCAL });
approx('full kv @8k = 1 GiB', kv.total / GIB, 1.0);

// Weights: 8e9 params x 4.5 bits / 8 = 4.5e9 B = 4.190951585769653 GiB.
let mem = calc.computeMemory({ model: FULL, bpw: 4.5, kvBytes: 2, context: 8192, parallel: 1, engine: ENG_LOCAL });
approx('weights 8B @4.5 bpw in GiB', mem.weights, 4.5e9 / GIB);
approx('total = weights + kv + overhead', mem.total, 4.5e9 / GIB + 1 + 1);

// Concurrent sequences multiply the KV cache, not the weights.
mem = calc.computeMemory({ model: FULL, bpw: 4.5, kvBytes: 2, context: 8192, parallel: 3, engine: ENG_LOCAL });
approx('parallel 3 -> kv 3 GiB', mem.kv, 3.0);
approx('parallel does not touch weights', mem.weights, 4.5e9 / GIB);

// q8_0 KV (8.5 bits = 1.0625 B/element) scales linearly: 1.0625 / 2 of fp16.
kv = calc.kvCache(FULL, { kvBytes: 1.0625, context: 8192, parallel: 1, engine: ENG_LOCAL });
approx('q8_0 kv @8k', kv.total / GIB, 0.53125);

// SWA at 32,768 tokens:
//   global: 40,960 x 2 x 32,768 = 2,684,354,560 B = 2.5 GiB
//   local, window-only: 409,600 x 2 x 1,024 = 838,860,800 B = 0.78125 GiB
kv = calc.kvCache(SWA, { kvBytes: 2, context: 32768, parallel: 1, engine: ENG_LOCAL });
approx('swa: global layers @32k', kv.full / GIB, 2.5);
approx('swa: local layers capped at window', kv.swa / GIB, 0.78125);
approx('swa: total', kv.total / GIB, 3.28125);
// An engine that does not store only the window pays the full context on every layer:
//   409,600 x 2 x 32,768 = 25 GiB, + 2.5 = 27.5 GiB.
kv = calc.kvCache(SWA, { kvBytes: 2, context: 32768, parallel: 1, engine: ENG_NOSWA });
approx('swa unaware engine pays full context', kv.total / GIB, 27.5);
// Below the window the cap does not bite: (40,960 + 409,600) x 2 x 512 = 461,373,440 B.
kv = calc.kvCache(SWA, { kvBytes: 2, context: 512, parallel: 1, engine: ENG_LOCAL });
approx('swa: context below window', kv.total / GIB, 461373440 / GIB);
// llama.cpp sizes the window cache as window + n_ubatch (512): 409,600 x 2 x 1,536 = 1.171875 GiB.
kv = calc.kvCache(SWA, { kvBytes: 2, context: 32768, parallel: 1, engine: Object.assign({}, ENG_LOCAL, { swaPadTokens: 512 }) });
approx('swa: window padded by the micro-batch', kv.swa / GIB, 1.171875);

// MLA: 35,136 x 2 B = 70,272 B/token; x 16,384 = 70,272 / 65,536 GiB = 1.072265625 GiB.
kv = calc.kvCache(MLA, { kvBytes: 2, context: 16384, parallel: 1, engine: ENG_LOCAL });
approx('mla latent kv @16k', kv.total / GIB, 1.072265625);

// Fixed recurrent state (Mamba / Gated DeltaNet layers): elements x state bytes per sequence,
// independent of context and of the KV dtype. 38,661,120 x 4 B = 154,644,480 B, x 2 sequences.
const SSM = Object.assign({}, FULL, { kvElemFull: 16384, stateElems: 38661120, stateBytes: 4, attention: 'ssm-hybrid' });
kv = calc.kvCache(SSM, { kvBytes: 1.0625, context: 4096, parallel: 2, engine: ENG_LOCAL });
approx('ssm state is per sequence and ignores kv dtype', kv.state, 2 * 38661120 * 4);
approx('ssm total = attention kv + state', kv.total, 2 * (16384 * 1.0625 * 4096) + 2 * 38661120 * 4);

// Memory budget.
const RTX = { id: 'rtx', kind: 'discrete', memGiB: 24, usableFraction: 1, bandwidth: 1008, backends: ['CUDA', 'Vulkan'] };
const MAC = { id: 'mac', kind: 'unified', memGiB: 128, usableFraction: 0.75, bandwidth: 546, backends: ['Metal', 'CPU'] };
const CPU = { id: 'cpu', kind: 'cpu', memGiB: 64, usableFraction: 1, reserveGiB: 4, bandwidth: 89.6, backends: ['CPU'] };
approx('budget: 2 x 24 GiB', calc.memoryBudget({ platform: RTX, gpus: 2, engine: ENG_LOCAL }), 48);
approx('budget: vLLM pre-allocates 92 %', calc.memoryBudget({ platform: RTX, gpus: 1, engine: ENG_VLLM }), 22.08);
approx('budget: unified memory uses the GPU fraction', calc.memoryBudget({ platform: MAC, gpus: 1, engine: ENG_LOCAL }), 96);
approx('budget: gpus ignored on unified memory', calc.memoryBudget({ platform: MAC, gpus: 4, engine: ENG_LOCAL }), 96);
approx('budget: CPU keeps a reserve for the OS', calc.memoryBudget({ platform: CPU, gpus: 1, engine: ENG_LOCAL }), 60);
// oMLX: min(RAM - 6 GiB, Metal cap); under 24 GiB of RAM the reserve is 4 GiB.
const ENG_OMLX = Object.assign({}, ENG_LOCAL, { policy: { type: 'ceiling', reserveGiB: 6, smallReserveGiB: 4, smallBelowGiB: 24 } });
approx('budget: RAM-minus-reserve binds when lower',
  calc.memoryBudget({ platform: Object.assign({}, MAC, { usableFraction: 1 }), gpus: 1, engine: ENG_OMLX }), 122);
approx('budget: Metal cap binds when lower', calc.memoryBudget({ platform: MAC, gpus: 1, engine: ENG_OMLX }), 96);
approx('budget: small machine reserves 4 GiB',
  calc.memoryBudget({ platform: Object.assign({}, MAC, { memGiB: 16, usableFraction: 1 }), gpus: 1, engine: ENG_OMLX }), 12);
// SGLang reserved_mem (MB) = 512 + 1.5 x chunked_prefill + tp/8 x 1024 + 2 x decode_max_bs.
//   24 GiB, tp 1: 512 + 1.5 x 2048 + 128 + 2 x 24 = 3,760 MB -> 24 - 3.671875 = 20.328125 GiB
//   80 GiB, tp 2: 512 + 1.5 x 8192 + 256 + 2 x 256 = 13,568 MB (above the 10 GiB floor)
//                 -> (80 - 13.25) x 2 = 133.5 GiB
const ENG_SGL = Object.assign({}, ENG_LOCAL, { policy: { type: 'sglang' } });
approx('budget: sglang reserve on 24 GiB', calc.memoryBudget({ platform: RTX, gpus: 1, engine: ENG_SGL }), 20.328125);
approx('budget: sglang reserve on 2 x 80 GiB',
  calc.memoryBudget({ platform: Object.assign({}, RTX, { memGiB: 80 }), gpus: 2, engine: ENG_SGL }), 133.5);
// 64 GiB card: 512 + 1.5 x 8192 + 128 + 2 x 32 ... max_bs is 256 at < 90 GiB:
//   512 + 12,288 + 128 + 512 = 13,440 MB = 13.125 GiB -> 50.875 GiB
approx('budget: sglang reserve on 64 GiB', calc.memoryBudget({ platform: Object.assign({}, RTX, { memGiB: 64 }), gpus: 1, engine: ENG_SGL }), 50.875);

// Max context. 24 GiB = 25,769,803,776 B; minus weights 4.5e9 and 1 GiB overhead
// leaves 20,196,061,952 B; / 131,072 B per token = 154,083.7 -> 154,083 tokens.
approx('max context, uncapped', calc.maxContext({ model: Object.assign({}, FULL, { maxContext: null }), bpw: 4.5,
  kvBytes: 2, parallel: 1, engine: ENG_LOCAL, budgetGiB: 24 }), 154083, 0);
approx('max context is capped by the model limit', calc.maxContext({ model: FULL, bpw: 4.5, kvBytes: 2, parallel: 1,
  engine: ENG_LOCAL, budgetGiB: 24 }), 131072, 0);
approx('max context is 0 when the weights do not fit', calc.maxContext({ model: FULL, bpw: 16, kvBytes: 2, parallel: 1,
  engine: ENG_LOCAL, budgetGiB: 12 }), 0, 0);
// SWA piecewise. weights 30e9 x 4.5 / 8 = 16,875,000,000 B; free = 24 GiB - weights - 1 GiB = 7,821,061,952 B.
//   Up to 1,024 tokens every layer grows: 450,560 x 2 x 1,024 = 922,746,880 B.
//   Remaining 6,898,315,072 B / (40,960 x 2) = 84,207.9 -> 1,024 + 84,207 = 85,231.
approx('max context with swa is piecewise', calc.maxContext({ model: SWA, bpw: 4.5, kvBytes: 2, parallel: 1,
  engine: ENG_LOCAL, budgetGiB: 24 }), 85231, 0);
// Without a window-only cache every layer grows: 7,821,061,952 / (450,560 x 2) = 8,679.3 -> 8,679.
approx('max context without a window-only cache', calc.maxContext({ model: SWA, bpw: 4.5, kvBytes: 2, parallel: 1,
  engine: ENG_NOSWA, budgetGiB: 24 }), 8679, 0);

// Decode speed: bandwidth 1,008 GB/s over bytes read per token.
//   active weights 8e9 x 4.5 / 8 = 4.5e9 B, plus one sequence's KV at 8k = 2^30 B
//   ceiling = 1,008e9 / 5,573,741,824 = 180.849... tok/s.
let sp = calc.decodeSpeed({ model: FULL, bpw: 4.5, kvBytes: 2, context: 8192, platform: RTX, engine: ENG_LOCAL, k: [0.5, 0.7] });
approx('decode ceiling', sp.ceiling, 1008e9 / (4.5e9 + GIB), 1e-6);
approx('decode low', sp.low, 0.5 * 1008e9 / (4.5e9 + GIB), 1e-6);
approx('decode high', sp.high, 0.7 * 1008e9 / (4.5e9 + GIB), 1e-6);
// MoE reads only the active experts: 37e9 x 4.5 / 8 for the MLA fixture.
sp = calc.decodeSpeed({ model: MLA, bpw: 4.5, kvBytes: 2, context: 16384, platform: MAC, engine: ENG_LOCAL, k: [1, 1] });
approx('moe decode uses active params', sp.ceiling, 546e9 / (37e9 * 4.5 / 8 + 70272 * 16384), 1e-6);
// Calibration picks the efficiency band by ceiling (measured k falls as the ceiling rises).
const CAL = { cpu: [0.67, 0.767], bands: [{ max: 30, k: [0.822, 0.876] }, { max: 60, k: [0.719, 0.805] },
              { max: 120, k: [0.585, 0.78] }, { max: 250, k: [0.435, 0.717] }, { max: null, k: [0.568, 0.72] }] };
assert('k band: 180 tok/s ceiling -> 120-250 band', JSON.stringify(calc.kRange(CAL, 'CUDA', 180)) === '[0.435,0.717]');
assert('k band: 25 tok/s ceiling -> 0-30 band', JSON.stringify(calc.kRange(CAL, 'Metal', 25)) === '[0.822,0.876]');
assert('k band: open-ended top band', JSON.stringify(calc.kRange(CAL, 'Vulkan', 900)) === '[0.568,0.72]');
assert('k band: CPU uses its own range', JSON.stringify(calc.kRange(CAL, 'CPU', 180)) === '[0.67,0.767]');
sp = calc.decodeSpeed({ model: FULL, bpw: 4.5, kvBytes: 2, context: 8192, platform: RTX, engine: ENG_LOCAL, calibration: CAL, backend: 'CUDA' });
approx('decode with calibration uses the band of its ceiling', sp.low, 0.435 * 1008e9 / (4.5e9 + GIB), 1e-6);
assert('dense model inside the calibrated range is not flagged', !sp.capped && !sp.uncalibrated);
// Past the highest calibrated ceiling measured speed stops scaling (compute-bound), so the
// ceiling the factor applies to is clamped. A 1B-active model on 1,792 GB/s: raw ceiling
// 1,792e9 / (1e9 x 4.5 / 8) = 3,185.8 tok/s, clamped to 523.2 x the top band.
const CALCAP = Object.assign({}, CAL, { maxCeiling: 523.2 });
const TINY = Object.assign({}, FULL, { params: 1, active: 1, kvElemFull: 0 });
sp = calc.decodeSpeed({ model: TINY, bpw: 4.5, kvBytes: 2, context: 1, platform: Object.assign({}, RTX, { bandwidth: 1792 }),
  engine: ENG_LOCAL, calibration: CALCAP, backend: 'CUDA' });
assert('ceiling past the calibrated range is capped', sp.capped === true);
approx('capped decode low', sp.low, 523.2 * 0.568, 1e-9);
approx('capped decode high', sp.high, 523.2 * 0.72, 1e-9);
// MoE without MoE calibration is flagged as uncalibrated.
sp = calc.decodeSpeed({ model: Object.assign({}, MLA, { moe: true }), bpw: 4.5, kvBytes: 2, context: 1024, platform: MAC,
  engine: ENG_LOCAL, calibration: CALCAP, backend: 'Metal' });
assert('moe without moe calibration is flagged', sp.uncalibrated === true);
// With a MoE calibration the MoE uses its own bands and is no longer flagged.
// MLA fixture on the Mac: ceiling = 546e9 / (37e9 x 4.5 / 8 + 70,272 x 1,024) = 26.1 tok/s -> "<100" band.
const CALMOE = Object.assign({}, CALCAP, { moe: { cpu: [0.38, 0.636], maxCeiling: 900,
  bands: [{ max: 100, k: [0.467, 0.591] }, { max: 250, k: [0.375, 0.481] }, { max: 500, k: [0.295, 0.448] }, { max: null, k: [0.3, 0.408] }] } });
sp = calc.decodeSpeed({ model: Object.assign({}, MLA, { moe: true }), bpw: 4.5, kvBytes: 2, context: 1024, platform: MAC,
  engine: ENG_LOCAL, calibration: CALMOE, backend: 'Metal' });
assert('moe with moe calibration is not flagged', sp.uncalibrated === false);
approx('moe uses the moe band', sp.low, 0.467 * 546e9 / (37e9 * 4.5 / 8 + 70272 * 1024), 1e-9);
// A dense model ignores the MoE calibration.
sp = calc.decodeSpeed({ model: FULL, bpw: 4.5, kvBytes: 2, context: 8192, platform: RTX, engine: ENG_LOCAL, calibration: CALMOE, backend: 'CUDA' });
approx('dense model keeps the dense band', sp.low, 0.435 * 1008e9 / (4.5e9 + GIB), 1e-6);

// Compatibility reasons. An engine flag of null means "undocumented": not blocked.
const LINEAR = Object.assign({}, FULL, { attention: 'linear-hybrid' });
assert('compatible combination has no reasons', calc.compatibility(ENG_LOCAL, RTX, FULL).length === 0);
assert('engine without the platform backend is flagged',
  calc.compatibility(ENG_VLLM, MAC, FULL).indexOf('backend') !== -1);
assert('engine without MLA is flagged for an MLA model',
  calc.compatibility(ENG_VLLM, RTX, MLA).indexOf('mla') !== -1);
assert('engine without hybrid support is flagged',
  calc.compatibility(Object.assign({}, ENG_LOCAL, { linearHybrid: false }), RTX, LINEAR).indexOf('linear-hybrid') !== -1);
assert('undocumented support (null) is not flagged',
  calc.compatibility(Object.assign({}, ENG_LOCAL, { linearHybrid: null }), RTX, LINEAR).length === 0);
assert('undocumented swa support is costed as full context',
  calc.kvCache(SWA, { kvBytes: 2, context: 32768, parallel: 1, engine: Object.assign({}, ENG_LOCAL, { swaWindowOnly: null }) }).total / GIB === 27.5);

// ---------------------------------------------------------------- Colibri
// Expert streaming: the fit is RAM against the project's documented minimum and free
// disk against the model's size; the VRAM only accelerates. Cold ceiling = SSD read
// speed / routed-expert GB read per token when every expert misses.
const GLM = { id: 'g', diskGB: 372, ramMinGB: 16, ramComfortableGB: 24, expertGBPerToken: 12.7 };
let cf = calc.colibriFit(GLM, { ramGB: 32, diskGB: 1000, diskGBps: 7 });
assert('colibri: 32 GB RAM + 1 TB disk fits GLM-5.2', cf.fits && cf.reasons.length === 0 && cf.comfortable);
approx('colibri: cold ceiling 7 GB/s / 12.7 GB', cf.coldTokps, 7 / 12.7, 1e-9);
cf = calc.colibriFit(GLM, { ramGB: 20, diskGB: 1000, diskGBps: 7 });
assert('colibri: between min and comfortable RAM fits but is not comfortable', cf.fits && !cf.comfortable);
cf = calc.colibriFit(GLM, { ramGB: 12, diskGB: 256, diskGBps: 3.5 });
assert('colibri: too little RAM and disk gives both reasons',
  !cf.fits && cf.reasons.indexOf('ram') !== -1 && cf.reasons.indexOf('disk') !== -1);
cf = calc.colibriFit(Object.assign({}, GLM, { expertGBPerToken: null }), { ramGB: 32, diskGB: 1000, diskGBps: 7 });
assert('colibri: no cold ceiling when the per-token read is undocumented', cf.coldTokps === null);
cf = calc.colibriFit(Object.assign({}, GLM, { diskGB: null, ramMinGB: null }), { ramGB: 32, diskGB: 1000, diskGBps: 7 });
assert('colibri: undocumented requirements are unknown, not a pass', cf.fits === null && cf.reasons.indexOf('unknown') !== -1);

// ------------------------------------------------------------------- cost
// Unchanged from v1. At 40 tok/s + 450 W the marginal electricity is close to a
// €0.6/M API, so break-even sits high (~572M tokens/mo), a real insight the tool shows.
const hi = calc.computeCost({ monthlyTokens: 1000e6, apiPrice: 0.6, throughput: 40, power: 450, elecPrice: 0.15, hwCost: 1800, amortMonths: 24 });
assert('cost: hwMonthly = 75', Math.abs(hi.hwMonthly - 75) < 0.01);
assert('cost: high volume self cheaper', hi.selfMonthly < hi.apiMonthly);
assert('cost: break-even finite & positive', isFinite(hi.breakeven) && hi.breakeven > 0);
approx('cost: break-even ~572M tokens', hi.breakeven / 1e6, 572, 5);
const lo = calc.computeCost({ monthlyTokens: 2e6, apiPrice: 0.6, throughput: 40, power: 450, elecPrice: 0.15, hwCost: 1800, amortMonths: 24 });
assert('cost: low volume API cheaper', lo.apiMonthly < lo.selfMonthly);

// --------------------------------------------------------- B. shipped data
if (calc.MODELS.length) {
  const M = id => calc.MODELS.find(m => m.id === id);
  const E = id => calc.ENGINES.find(e => e.id === id);
  const P = id => calc.PLATFORMS.find(p => p.id === id);
  assert('data: 40+ models', calc.MODELS.length >= 40);
  assert('data: 8 engines + Colibri', calc.ENGINES.length === 9 && calc.ENGINES[8].id === 'colibri');
  assert('data: Colibri GLM-5.2 needs 372 GB disk and 16 GB RAM', calc.COLIBRI.models.find(m => m.id === 'coli-glm52').diskGB === 372 && calc.COLIBRI.models.find(m => m.id === 'coli-glm52').ramMinGB === 16);
  assert('data: 30+ platform variants', calc.PLATFORMS.length >= 30);
  assert('data: ids are unique', new Set(calc.MODELS.map(m => m.id)).size === calc.MODELS.length
    && new Set(calc.PLATFORMS.map(p => p.id)).size === calc.PLATFORMS.length);
  // Gemma 3 27B: 10 global + 52 local layers, 16 KV heads x 128 (config.json), window 1,024.
  approx('data: gemma3-27b global elements', M('gemma3-27b').kvElemFull, 2 * 10 * 16 * 128, 0);
  approx('data: gemma3-27b local elements', M('gemma3-27b').kvElemSwa, 2 * 52 * 16 * 128, 0);
  // Qwen3 32B keeps its explicit head_dim 128 (hidden/heads would give 80).
  approx('data: qwen3-32b elements use head_dim 128', M('qwen3-32b').kvElemFull, 2 * 64 * 8 * 128, 0);
  // Qwen3.6 35B-A3B: only 10 of 40 layers carry KV, plus a fixed Gated DeltaNet state.
  approx('data: qwen36-35b 10 kv layers', M('qwen36-35b').kvElemFull, 2 * 10 * 2 * 256, 0);
  assert('data: qwen36-35b has a fixed state', M('qwen36-35b').stateElems > 0);
  // DeepSeek-V3.1 MLA: 61 x (512 + 64).
  approx('data: deepseek-v31 mla elements', M('deepseek-v31').kvElemFull, 61 * 576, 0);
  approx('data: vLLM utilization default 0.92', E('vllm').policy.fraction, 0.92, 0);
  approx('data: GGUF Q4_K_M 4.8944 bpw', E('llamacpp').formats.find(f => f.id === 'gguf-q4_k_m').bpw, 4.8944, 0);
  approx('data: RTX 4090 bandwidth 1008 GB/s', P('rtx4090').bandwidth, 1008, 0);
  // Every chat model runs somewhere with llama.cpp (it supports every attention kind here).
  assert('data: llama.cpp supports every chat model on a 4090',
    calc.MODELS.filter(m => !m.decision).every(m => calc.compatibility(E('llamacpp'), P('rtx4090'), m).length === 0));
  // Decision models (2026-11-12): the engine must run the decision head or read-out.
  const dm = calc.MODELS.filter(m => m.decision);
  assert('data: 14 decision models', dm.length === 14);
  assert('data: Clef runs on llama.cpp (arch clef, PR #29831)', calc.decisionSupport(E('llamacpp'), M('clef-27b')) === 'yes');
  assert('data: Clef 27B is not in vLLM (only clef-flash and clef-omni are)',
    calc.compatibility(E('vllm'), P('rtx4090'), M('clef-27b')).indexOf('decision') !== -1 &&
    calc.compatibility(E('vllm'), P('rtx4090'), M('clef-flash-9b')).length === 0);
  assert('data: LM Studio gives meaningless Clef output', calc.decisionSupport(E('lmstudio'), M('clef-27b')) === 'no');
  assert('data: OpenJev on vLLM needs its helper shim', calc.decisionSupport(E('vllm'), M('openjev-27b')) === 'helper');
  assert('data: OpenJev is non-commercial (CC BY-NC 4.0)', M('openjev-27b').decision.commercial === false);
  // Clef ships its 0.13B joint head apart from the backbone safetensors total.
  approx('data: Clef params = 27.36 backbone + 0.13 head', M('clef-27b').params, 27.49, 1e-9);
  // Same backbone as Qwen3.8-27B, so the same KV per token.
  approx('data: Clef KV = Qwen3.8-27B KV', M('clef-27b').kvElemFull, M('qwen38-27b').kvElemFull, 0);
  assert('data: chat models carry no decision block', !M('qwen38-27b').decision);
  // Voice (TTS), 2026-11-12: published memory wins over the 16-bit estimate, per platform kind.
  const T = id => calc.TTS.find(m => m.id === id);
  assert('data: 39 TTS models', calc.TTS.length === 39);
  let tf = calc.ttsFit(T('dia-1.6b'), P('rtx4090'));
  assert('tts: Dia on a 4090 uses the published 4.4 GB and fits', !tf.estimated && tf.needGB === 4.4 && tf.fits === true && tf.runsOn === 'gpu');
  // Kokoro has no published memory: 0.082B x 2 B = 0.164e9 B = 0.15 GiB estimate.
  tf = calc.ttsFit(T('kokoro-82m'), P('rtx4090'));
  assert('tts: Kokoro is a flagged 16-bit estimate', tf.estimated && Math.abs(tf.needGB - 0.15) < 0.01);
  // Drama 3: 48 GB FP8 on a discrete card (does not fit 24 GB), 26 GB MLX 4-bit on a Mac.
  tf = calc.ttsFit(T('fish-audio-drama-3'), P('rtx4090'));
  assert('tts: Drama 3 needs 48 GB on a discrete GPU', tf.needGB === 48 && tf.fits === false);
  const mac = calc.PLATFORMS.find(p => p.kind === 'unified' && p.vendor === 'Apple' && p.memGiB === 64);
  tf = calc.ttsFit(T('fish-audio-drama-3'), mac);
  assert('tts: Drama 3 uses the 26 GB MLX figure on a 64 GB Mac and fits', tf.needGB === 26 && tf.fits === true);
  // VibeVoice Realtime publishes only CUDA: no runtime on a Mac.
  assert('tts: CUDA-only VibeVoice Realtime has no runtime on a Mac', calc.ttsFit(T('vibevoice-realtime-0.5b'), mac).runsOn === null);
  assert('tts: every model has a Spanish class', calc.TTS.every(m => ['yes', 'variant', 'experimental', 'no'].indexOf(m.spanish) !== -1));
  assert('tts: Voxtral is non-commercial (CC BY-NC)', T('voxtral-4b-tts-2603').commercial === false);
  // vLLM has no Metal backend (only a community plugin), so it must not "fit" a Mac via CPU.
  assert('data: vLLM is incompatible with an Apple Silicon Mac',
    calc.compatibility(E('vllm'), P('m4max-40c-128'), M('qwen3-8b')).indexOf('backend') !== -1);
  assert('data: oMLX runs only on Apple Silicon',
    calc.compatibility(E('omlx'), P('rtx4090'), M('qwen3-8b')).indexOf('backend') !== -1
    && calc.compatibility(E('omlx'), P('m4max-40c-128'), M('qwen3-8b')).length === 0);
  // Regression on a real case: Llama 3.1 8B, GGUF Q4_K_M, f16 KV @8k on a 24 GB card fits.
  const x = { model: M('llama8b'), bpw: 4.8944, kvBytes: 2, context: 8192, parallel: 1, engine: E('llamacpp') };
  assert('data: llama8b q4_k_m @8k fits a 4090', calc.computeMemory(x).total <= calc.memoryBudget({ platform: P('rtx4090'), gpus: 1, engine: E('llamacpp') }));
}

console.log(failures === 0 ? '\nALL PASS' : `\n${failures} FAILURES`);
process.exit(failures === 0 ? 0 : 1);
