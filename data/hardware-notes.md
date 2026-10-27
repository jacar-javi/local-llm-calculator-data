# hardware.json: research notes (2026-09-15)

Companion to `hardware.json`. Every number was taken from a page fetched on 2026-09-15: vendor spec pages, datasheets and newsroom posts, official docs, or llama.cpp GitHub threads. Where a figure could not be verified, the field is `null` and the reason is below. `price_eur` is `null` everywhere. `official_price` is filled only with vendor-published prices: the Apple Spain store, NVIDIA's own es-es price list, or a launch MSRP from a vendor press release or spec page.

## 1. Formulas

| Quantity | Formula | Check against a vendor figure |
|---|---|---|
| GDDR bandwidth | bus width (bits) x speed (Gbps) / 8 | RX 7900 XTX: 384 x 20 / 8 = 960, and AMD lists "Up to 960 GB/s". Arc B580: 192 x 19 / 8 = 456, and Intel lists "456 GB/s". |
| HBM / LPDDR bandwidth (NVIDIA datasheets) | bus width x clock MHz x 2 / 8 | A100: 5120 x 1512 x 2 / 8 = 1935, and NVIDIA lists "1,935 GB/s". DGX Spark: 256 x 4266 x 2 / 8 = 273, and NVIDIA lists "273 GB/s". |
| DDR / LPDDR system memory | channels x MT/s x 8 bytes | EPYC 9004: 12 x 4800 x 8 = 460.8 GB/s, and amd.com lists "Per Socket Mem BW 460.8 GB/s". Strix Halo: 256-bit x 8000 / 8 = 256, and AMD blogs say "256 GB/s". |
| Decode ceiling | bandwidth_GBps / model_size_GB | llama-bench reports GiB, so size_GB = GiB x 1.073741824. |
| Efficiency | k = measured tg128 / ceiling | |

## 2. Bandwidth: vendor-published vs computed

| id | GB/s | source | notes |
|---|---|---|---|
| rtx3060-12 | 360 | computed | 192-bit (NVIDIA) x 15 Gbps (ASUS partner page) |
| rtx3090 | 936 | computed | 384-bit (NVIDIA) x 19.5 Gbps (ASUS) |
| rtx4060ti-16 | 288 | computed | 128-bit (NVIDIA) x 18 Gbps (ASUS) |
| rtx4070tisuper | 672 | computed | 256-bit (NVIDIA) x 21 Gbps (ASUS) |
| rtx4090 | 1008 | computed | 384-bit (NVIDIA) x 21 Gbps (ASUS) |
| rtx5060ti-16 / 5070ti / 5080 / 5090 | 448 / 896 / 960 / 1792 | vendor | nvidia.com compare page "Memory Bandwidth" row. The 5070 Ti and 5080 match ASUS speeds (28 and 30 Gbps). |
| rtxpro6000-blackwell | 1792 | vendor | Workstation and Max-Q datasheets. **The Server Edition page says 1597 GB/s**, with no reason given. |
| a100-80 | 1935 | vendor | PCIe card. SXM is 2,039. The product brief rounds to "Up to 1.94 TB/s". |
| h100-80 | 2000 | vendor | PCIe brief gives "2,000 GB/s", but brief inputs compute to 2,039. **The nvidia.com H100 page lists the 80GB SXM at "3.35TB/s".** That equals the H100 NVL's 2,619 MHz HBM3 clock on a 5120-bit bus, so it looks like a copy error. Not used. |
| h200 | 4813 | vendor | NVL brief. The web page says "4.8TB/s" and "Preliminary specifications". |
| rx7900xtx / rx9070xt | 960 / 640 | vendor | Both match the computed value. **Ignore the 7900 XTX page's "Effective Memory Bandwidth Up to 3500 GB/s"** (marketing figure). |
| radeon-ai-pro-r9700 | 640 | vendor | "Peak Memory Bandwidth 640 GB/s", 256-bit. AMD lists no memory speed, so there is no computed cross-check. |
| arc-b580 / arc-pro-b60 | 456 / 456 | vendor | Both match 192-bit x 19 Gbps. |
| dgx-spark | 273 | vendor | NVIDIA docs: "256-bit interface, 4266 MHz, 273 GB/s", "16 channels (256 bit) LPDDR5X 8533". |
| strix-halo-395-128 | 256 | vendor (AMD blog) | Spec page: "256-bit LPDDR5x", "LPDDR5x-8000". Real systems may run slower memory. One unverified #10879 comment reports a measured 230 GB/s. |
| Apple (all) | 120 to 1200 | vendor | apple.com and support.apple.com tech specs. M5 Ultra is given as "1.2TB/s" (rounded, stored as 1200). **M6 differs by RAM:** 153 GB/s at 16 GB, 170 GB/s at 24/32 GB, so it is split into two entries. |
| cpu-ddr5-5600-2ch | 89.6 | computed | 2 x 5600 x 8. **AMD's 9950X page drops to DDR5-3600 (57.6 GB/s) with 4 DIMMs.** |
| cpu-epyc9004-12ch | 460.8 | vendor | amd.com, per socket, every channel populated. |

## 3. Memory the GPU can actually use

### Apple Silicon (`gpu_usable_memory.rule = macos_recommendedMaxWorkingSetSize`)

**How engines use the limit**
- Apple documents the API but gives no fraction. The Metal doc defines `recommendedMaxWorkingSetSize` as "An approximation of how much memory, in bytes, this GPU device can allocate without affecting its runtime performance."
- llama.cpp reports that value as the Metal device's total memory. `ggml-metal-device.m`: `*total = dev->mtl_device.recommendedMaxWorkingSetSize;`, with the comment "it's possible to allocate more than `recommendedMaxWorkingSetSize`". It logs the value as `/ 1e6` "MB", i.e. decimal megabytes.

**Default rule, macOS 13-15**
- **The GPU gets 2/3 of RAM up to 32 GiB, and 3/4 above that.**
- Source: the decompiled kernel routine posted in llama.cpp discussion #2182 (dr3murr, 2023-07-13). Quote: `float reservePercent = 33.333f; // Use a lower percentage if the unified memory is larger than 32 GB if (unifiedMemoryBytes > 0x800000000LL) { reservePercent = 25.0f; }`
- Logs agree: 16 GB M1 → 10922.67 MiB (0.667); 36 GB M3 Pro → 28991.03 MB (0.750); 64 GB M1 Max on Sonoma → 49152 MiB (0.75).
- ggerganov in #4167: "By default, you can use ~75% of the total RAM with the GPU."

**macOS 26 (Tahoe) reads higher.** This is observed in logs, not documented, so it is stored as `macos_26_observed_fraction_range: [0.74, 0.81]`.
- llama.cpp issue #20141, M4 Pro 48 GB on Tahoe 26.3: `recommendedMaxWorkingSetSize = 40200.90 MB`, which is 0.78. I verified this one myself.
- #4167 comment 18210351, M5 16 GB on macOS 26.6.1: "model weights (13.5 GiB) exceed recommendedMaxWorkingSetSize (12.7 GB)". That is about 0.74 if the units are decimal. I verified this one myself.
- Also reported, but not re-fetched by me: 64 GB M1 Max on 26.4 → 0.81 (LM Studio bug tracker #1728), and 32 GiB M2 Max on 26.5.2 → 78% (vendor blog).

**Recommendation.** Default to the 2/3 or 3/4 rule, which is conservative and under-states macOS 26 by about 1-4 GB. Offer a "macOS 26" toggle or the sysctl override.

**Override.** `sudo sysctl iogpu.wired_limit_mb=N`. It is not persistent across reboot; kdyke in #2182 notes "You'd have to do it every boot as it's not sticky". The MLX-LM README (macOS 15+) says "The value `N` should be larger than the size of the model in megabytes but smaller than the memory size of the machine." Another #2182 reply warns: "I would not recommend going to 100% as the OS needs some reasonable amount of memory".

**MLX** allows allocations up to `min(1.5 x max_rec_size, 0.95 x memsize)` (MLX `allocator.cpp`), at a speed cost.

### AMD Strix Halo, Ryzen AI Max+ 395 (`rule = os-dependent`)

**Windows**
- AMD Variable Graphics Memory, set in the BIOS or Adrenalin. AMD: "up to 96GB can be converted to VRAM through AMD Variable Graphics Memory" (amd.com blog).
- The AMD VGM FAQ: "A 128GB system by default would be set to 512 MB discrete graphics memory ... 64.3 total graphics addressable memory". Also: "The highest performance can be unlocked by limiting workloads to 96GB – but if needed – the iGPU (set to 96GB VGM) can technically access a total graphics memory size of 112GB".
- The 64 GB systems allow 48 GB of VGM. `fraction: 0.75` comes from those two stated cases; AMD publishes no formula.
- The full 96 GB for Vulkan llama.cpp arrived with Adrenalin 25.8.1.

**Linux**
- GPU memory is GTT-backed. ROCm docs: "By default, the GTT limit is set to approximately 50 percent of total system RAM."
- It can be raised via `/sys/module/ttm/parameters/pages_limit` (`amd-ttm --set 100` → "Successfully set TTM pages limit to 26214400 pages (100.00 GB)" on a machine showing "Total system memory: 125.65 GB").
- AMD recommends keeping the BIOS carve-out small ("for example, 0.5 GB").
- AMD states no Linux maximum.
- Kernel minimums for the fixes: Ubuntu 24.04 HWE 6.17.0-19.19~24.04.2, OEM 6.14.0-1018, other distros 6.18.4.
- ROCm on Windows covers PyTorch only: "Only Pytorch is currently available on Windows - the rest of the ROCm stack is only supported on Linux."

### NVIDIA DGX Spark (`rule = dynamic-uma`, `fraction = null`)

**What NVIDIA says**
- There is no fixed carve-out. NVIDIA porting guide: "Unlike other vendors, which use a fixed carve-out for the iGPU in the shared memory, we utilize a dynamic unified memory architecture (UMA), allowing both the CPU and iGPU to access the same memory space."
- The GPU can allocate what Linux has free.
- NVIDIA known issues: "the memory size reported by cudaMemGetInfo may be smaller than the actual allocatable memory". Also: nvidia-smi shows "Memory-Usage: Not Supported".

**Community evidence (NVIDIA forum)**
- `free` shows **121 GiB** total on the Founders Edition and **119-119.7 GiB** on some OEM firmware.
- One reply says a BIOS update "frees up 2GB".
- A claimed 96 GB GPU cap was answered by a community member (not NVIDIA staff): "Not true, there's no such limit anywhere."

**Calculator model:** usable GPU ≈ OS-visible total (~119.7-121.7 GiB) minus OS and apps. NVIDIA's FAQ suggests dropping page cache if an allocation fails.

**Power:** GB10 TDP is 140 W (CPU+GPU); peak system power is 240 W.

## 4. Speed calibration

- **Points:** 132 points were collected; 129 are used in stats (3 are flagged `exclude_from_stats`).
  - Metal: 94 points. 75 are rows of the #4167 summary table (build 8e672ef, Nov 2023). 19 are recent-build comments or rows (2026).
  - CUDA: 13. Vulkan: 12. ROCm: 5. CPU: 5 used, plus 3 excluded.
- **Model coverage:** most GPU points are Llama-2-7B Q4_0, the scoreboards' standard model. Larger dense points: Qwen3-32B Q4_K_M on the R9700 (Vulkan, ROCm), Llama-3.1-70B F16 on EPYC, and Llama-7B F16/Q8_0 on Apple.

**Headline k (all used points)**

| Backend | median | p25 | p75 | n |
|---|---|---|---|---|
| Metal | 0.772 | 0.658 | 0.834 | 94 |
| CUDA | 0.725 | 0.619 | 0.776 | 13 |
| Vulkan | 0.768 | 0.586 | 0.819 | 12 |
| ROCm | 0.665 | 0.605 | 0.742 | 5 |
| CPU (single socket, llama-bench) | 0.728 | 0.670 | 0.767 | 5 |

**k falls as the ceiling rises.** A small model on fast memory becomes compute-bound. GPU backends pooled:

| Ceiling (tok/s) | median k | p25-p75 | n |
|---|---|---|---|
| < 30 | 0.843 | 0.822-0.876 | 32 |
| 30-60 | 0.768 | 0.719-0.805 | 32 |
| 60-120 | 0.695 | 0.585-0.780 | 32 |
| 120-250 | 0.581 | 0.435-0.717 | 18 |
| ≥ 250 | 0.642 | 0.568-0.720 | 10 |

More subsets are in `calibration.k_subsets`:
- **Metal by build:** the 2023 build gives 0.760 (n=75); the 2026 builds give 0.814 (n=19). The same M1 Ultra 64-core F16 run goes from 37.01 to 47.82 tok/s. The M2 Ultra time-series row goes from 41.02 (8e672ef) to 49.55 (c1d0e7a + FA).
- **Metal by quant:** F16 0.833, Q8_0 0.793, Q4_0 0.687.
- **Bandwidth ≤ 1000 GB/s:** CUDA 0.766, Vulkan 0.798.
- **Outliers:** the A100 (0.377) and H100 (0.512) on 7B Q4_0 are far below bandwidth-bound.

**Recommendation for the tool.** Choose k by the ceiling of the model being estimated, not one scalar per backend.
- Large models fall in the < 60 tok/s range, where Metal is 0.81 (IQR 0.77-0.85, n=61) and pooled GPU is about 0.77-0.84.
- CUDA, Vulkan and ROCm each have only one point below 60 tok/s. Use the pooled GPU bands for them. On the same AMD hardware, ROCm k / Vulkan k is 0.645-0.934 (median 0.85, n=5: 7900 XTX 0.92, 9070 XT 0.74, R9700 7B 0.65, R9700 32B 0.85, Strix Halo 0.93). If the tool models a backend penalty, ROCm is the one that needs it.
- **CPU:** 0.73, but n=5 and 4 of the 5 come from one author's modified build.

**Excluded from stats**
- Two dual-socket EPYC runs: the calculator entry is per socket.
- A Ryzen AI 9 HX 370 figure: it comes from Ollama, not llama-bench, and its size was rounded.

**Included with a caveat:** the DGX Spark Qwen2.5-Coder-7B Q8_0 point is tg32 with FA on (from `benches/dgx-spark/dgx-spark.md`), not tg128.

**Missing coverage**
- **CPU desktop:** no dual-channel DDR5 desktop llama-bench tg128 result was found for a dense model. The only dual-channel point is a DDR4-3200 laptop (k=0.52).
- **Base M1:** Apple publishes no bandwidth figure, so it is skipped.
- **Intel Arc:** no SYCL points collected; Vulkan only.
- **H200:** no calibration point.

## 5. Skipped or unverifiable items

- **Power `null`:**
  - Apple M4 Max 40c, and every M5 and M6. Apple only measured max power for M4 (65 W), M4 Pro (140 W), M4 Max 32c (145 W) and M3 Ultra 32C/80G (270 W). The 2026 spec pages give only "Maximum continuous power" supply ratings (Mac mini 155 W, Mac Studio 480 W).
  - EPYC 9004 entry: TDP depends on SKU, 200 W for the 9124 and 360 W for the 9654.
- **Other power caveats:**
  - RTX 4060 Ti: NVIDIA says "165 or 160" W without saying which config draws which; 165 W is used.
  - Strix Halo: 120 W is the top of the cTDP range (45-120 W, default 55 W), set by the OEM.
  - DGX Spark: 140 W is the chip TDP.
- **Memory type:** Apple never states the LPDDR generation.
- **Backends:**
  - NVIDIA does not state Vulkan for the A100, H100, H200 or GB10 (DGX Spark), so those list CUDA only. GB10 does run llama.cpp Vulkan (#10879 scoreboard).
  - Arc Pro B60 SYCL rests on Intel's "oneAPI Support: Yes"; llama.cpp's SYCL doc verifies only the B580.
  - llama.cpp's AMD path is now named "HIP" and "Make sure to have ROCm installed", so it is listed as ROCm.
- **Apple M3 Ultra 512 GB:** Apple's current tech-specs page lists 96/256 GB. 512 GB is kept because it was offered at launch (newsroom) and Apple used it for the power figure.
- **Beyond the requested list:** Apple shipped M5 Ultra and M6 (announced 2026-08-25), so they are added.
- **M5 Pro and M5 Max exist.** M5 Pro/Max were announced 2026-03-03.

## 6. Official prices (`official_price`)

- **NVIDIA EUR prices:**
  - Source: `pricelist.js`, which the es-es product pages load.
  - Values: 4070 Ti SUPER "A partir de 899 €", 4090 "PVPR 1.779 €*", 5070 Ti "PVPR 889 €*", 5080 "PVPR 1.059 €*", 5090 "PVPR 2.099 €*".
  - Verified by fetching `pricelist.js`. I did not check them in a rendered browser page.
  - These are NVIDIA's current recommended prices; the 4090's is not its launch price.
- **NVIDIA USD launch MSRPs:** 3060 $329, 3090 $1,499, 4060 Ti 16 GB $499, 5060 Ti 16 GB $429. The es-es "405 €" is the 8 GB card.
- **DGX Spark:** $4,699, NVIDIA's current MSRP. It was raised on 2026-02-23 from $3,999, per an NVIDIA employee on NVIDIA's forum. The kind is "launch-msrp" because the schema has no "current" kind. The Spanish NVIDIA marketplace could not be reached from here.
- **AMD and Intel:** RX 7900 XTX $999 SEP, RX 9070 XT $599 SEP, Arc B580 $249 "Recommended Customer Price".
- **No official price, so `null`:** R9700 ($1,299 appears only in third-party coverage), Arc Pro B60, Strix Halo (sold by OEMs), RTX PRO 6000, A100, H100, H200.
- **Apple:**
  - `official_price` is a **list**, one entry per verifiable config, from the JSON-LD on apple.com/es config pages. Three were re-checked: M5 Ultra 96GB 6,649 €, M6 16GB 1,069 €, M5 Max 36GB 3,029 €.
  - M4, M4 Pro, M4 Max and M3 Ultra are no longer sold there, so they are `null`.
  - Some RAM tiers exist only in the configurator (for example M5 Max 128 GB, M5 Ultra 256/512 GB), so they have no price.
  - Where the cheapest machine has no standard config at that RAM, a MacBook Pro config is listed and labelled as such.

## 7. Caveats

- **Scoreboards are optimistic.** They keep each device's best run, mixing commits, drivers and operating systems. The same card can differ a lot by backend (RTX 5070 Ti: CUDA 0.755, Vulkan 0.579).
- **File size slightly overstates bytes read per token.** The embedding table is only row-indexed, and KV-cache reads add a little at tg128. k absorbs both.
- **Newer builds report a smaller file.** Llama-2-7B Q4_0 shows as 3.53 GiB instead of 3.56 GiB. The M2 Ultra c1d0e7a Q4_0 point uses 3.53 GiB, taken from other c1d0e7a runs in the same thread.
- **MoE models:** all calibration is dense. Use active-parameter bytes plus shared/attention weights. Note that the EPYC data shows poor dual-socket scaling for MoE decode.
- **Multi-GPU, speculative decoding and batching** are out of scope.
