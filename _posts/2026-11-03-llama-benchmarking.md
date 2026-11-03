
# Benchmarking a local LLM performance

Today, I'm benchmarking the local performance of a `Qwen3.8-27B` model running with `llama-server` on my home 
RTX 3090 (24GB) GPU. For LLM inference, I'm looking to understand the trade-offs for different settings on 
my specific hardware for single-stream, quality-first yet fast decode (output tokens-per-second, TPS) split across
both GPUs.

Using the built in `llama-bench` tool that comes with llama.cpp, I started from the full set of available options, 
summarized below


## Options under consideration

1. **Batch / context flags** &rarr; Bigger context = bigger KV cache = less headroom and more memory bandwidth pressure during 
long generations. General wisdom is to set a moderate constext size to keep KV small and improve decode TPS

2. **`-b (batchsize)` and `-ub (ubatch)`** &rarr; Increasing batch helps prompt-processing prefill time-to-first-token (TTFT)

3. **KV cache quantization** &rarr; Quantized KV cache reduces VRAM usage and can improve bandwidth-bound decode performance, 
especially with flash attention

4. **Flash attention** &rarr; is opaque to me, but seems to be a specific method to managing attention as KV cache size increases.
I wonder if that's because attention complexity is quadratic wrt sequence length :thinking: 

5. **GPU split control** &rarr; Ensures both GPUs are used evenly. The `llama-server` logs usually prints some "offloaded 
X / Y layers to GPU" message. Also `nvidia-smi` should show a `llama-server` sub-process on each GPU

6. **Thread count** &rarr; CPU threads matter for the offloaded parts and sampling

7. **`-np 1` number of parallel sequences = 1 (single user)** &rarr; Good for max single-stream TPS

8. **Sampler settings / quality**: Lots of unkowns about `presence_penalty` & `repeat_penalty` i.e. for say discouraging 
repeating tokens or improving coherence/quality wrt my generation goals (creative vs factual). All summer,
I've been successfully using the very basic settings below; however, sampling params should NOT affect TPS
```
--temp 1.0 --top-p 0.95 --min-p 0.0 --top-k 20  --presence_penalty 1.5 --repeat_penalty 1.0 
```
For Qwen models, is the typically recommended sampling is `temp ~0.6-0.7`, `top-p 0.9`? 

9. **`--jinja` flag** &rarr; enables the Jinja templating from the GGUF's chat template, which for Qwen is important to 
get system/user roles and think tags right.


## llama-bench baselines

Baseline performance for the 3 models under test, `pp` is prompt (input) processing, `tg` is token-generation

```text
$ ../build/bin/llama-bench  --model ./Qwen3.8-27B-UD-Q8_K_XL.gguf
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |           pp512 |      1384.94 ± 23.88 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |           tg128 |         26.44 ± 0.03 |

$ ../build/bin/llama-bench  --model ./Qwen3.8-27B-UD-Q6_K_XL.gguf
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |           pp512 |      1303.13 ± 14.21 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |           tg128 |         30.89 ± 0.10 |


$ ../build/bin/llama-bench  --model ./Qwen3.6-27B-UD-Q4_K_XL.gguf
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |           pp512 |      1359.90 ± 16.57 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |           tg128 |         41.94 ± 0.09 |
```

## Performance vs. Context Size
Prompt Processing (pp) of 512 tokens, @d5000 means how many tokens are in the kv cache (context cache)
So `pp512 @ d20000` means the test was measuring the speed of prompt processing 512 tokens with 20,000 tokens 
in the KV cache. 

```text
$ ../build/bin/llama-bench  --model ./Qwen3.6-27B-UD-Q4_K_XL.gguf --model ./Qwen3.8-27B-UD-Q6_K_XL.gguf --model ./Qwen3.8-27B-UD-Q8_K_XL.gguf --n-depth 0,5000,10000,20000,80000
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |           pp512 |      1365.06 ± 17.63 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |           tg128 |         41.96 ± 0.09 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |   pp512 @ d5000 |      1280.06 ± 15.97 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |   tg128 @ d5000 |         41.19 ± 0.02 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |  pp512 @ d10000 |       1212.49 ± 9.48 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |  tg128 @ d10000 |         40.75 ± 0.06 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |  pp512 @ d20000 |       1097.45 ± 8.15 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |  tg128 @ d20000 |         39.47 ± 0.05 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |  pp512 @ d80000 |        704.37 ± 4.76 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |  tg128 @ d80000 |         33.40 ± 0.05 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |           pp512 |      1275.58 ± 12.43 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |           tg128 |         30.85 ± 0.07 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |   pp512 @ d5000 |      1203.41 ± 10.92 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |   tg128 @ d5000 |         30.41 ± 0.02 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |  pp512 @ d10000 |       1143.50 ± 9.28 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |  tg128 @ d10000 |         30.14 ± 0.02 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |  pp512 @ d20000 |       1040.79 ± 7.99 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |  tg128 @ d20000 |         29.44 ± 0.04 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |  pp512 @ d80000 |        679.40 ± 3.27 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |  tg128 @ d80000 |         25.98 ± 0.01 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |           pp512 |      1348.27 ± 11.69 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |           tg128 |         26.42 ± 0.02 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |   pp512 @ d5000 |      1267.46 ± 19.82 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |   tg128 @ d5000 |         26.11 ± 0.02 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |  pp512 @ d10000 |      1196.74 ± 12.33 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |  tg128 @ d10000 |         26.01 ± 0.02 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |  pp512 @ d20000 |      1092.57 ± 10.98 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |  tg128 @ d20000 |         25.43 ± 0.03 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |  pp512 @ d80000 |        704.72 ± 6.63 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |  tg128 @ d80000 |         22.80 ± 0.03 |
```

The results are kinda verbose, but we can see that the baseline for our desired `Qwen3.8-27B` model goes from 26 to 22 TPS,
as the context grows


## Performance vs. Prompt size
This will only test prompt processing, no token generation, which is disabled with `--n-gen 0`

```text
$ ../build/bin/llama-bench  --model ./Qwen3.6-27B-UD-Q4_K_XL.gguf --model ./Qwen3.8-27B-UD-Q6_K_XL.gguf --model ./Qwen3.8-27B-UD-Q8_K_XL.gguf --n-prompt 1000,5000,10000-100000+10000 --n-gen 0
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |          pp1000 |       1682.91 ± 5.38 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |          pp5000 |       2133.96 ± 3.14 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |         pp10000 |       2167.79 ± 4.35 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |         pp20000 |       2084.95 ± 3.19 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |         pp30000 |       1997.84 ± 1.34 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |         pp40000 |       1909.53 ± 1.06 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |         pp50000 |       1830.98 ± 2.06 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |         pp60000 |       1749.64 ± 6.46 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |         pp70000 |       1682.07 ± 6.74 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |         pp80000 |       1618.40 ± 3.05 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |         pp90000 |       1561.62 ± 2.00 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |        pp100000 |       1506.30 ± 2.16 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |          pp1000 |       1535.71 ± 8.19 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |          pp5000 |       1972.98 ± 4.11 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |         pp10000 |       2000.52 ± 1.64 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |         pp20000 |      1933.96 ± 18.39 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |         pp30000 |      1860.38 ± 18.61 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |         pp40000 |       1794.18 ± 4.71 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |         pp50000 |       1723.99 ± 5.41 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |         pp60000 |       1656.21 ± 2.86 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |         pp70000 |       1599.01 ± 1.29 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |         pp80000 |       1541.58 ± 1.50 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |         pp90000 |       1484.38 ± 5.86 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |        pp100000 |       1435.11 ± 4.04 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |          pp1000 |       1630.57 ± 8.07 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |          pp5000 |       2112.90 ± 3.02 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |         pp10000 |       2143.04 ± 3.55 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |         pp20000 |       2074.41 ± 2.06 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |         pp30000 |       1985.64 ± 5.28 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |         pp40000 |       1904.37 ± 3.34 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |         pp50000 |       1817.31 ± 9.83 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |         pp60000 |       1747.18 ± 3.61 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |         pp70000 |       1687.10 ± 2.84 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |         pp80000 |       1618.29 ± 8.31 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |         pp90000 |       1562.71 ± 2.86 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |        pp100000 |       1508.52 ± 4.31 |
```
Everything is blazing fast, nothing to see here


## Performance vs. batch-size @ 5000 token Prompt size

```text
$ ../build/bin/llama-bench  --model ./Qwen3.6-27B-UD-Q4_K_XL.gguf --model ./Qwen3.8-27B-UD-Q6_K_XL.gguf --model ./Qwen3.8-27B-UD-Q8_K_XL.gguf --batch-size 256,512,1024,2048,4096,8192,10240 --n-prompt 5000 --n-gen 0
| model                          |       size |     params | backend    | ngl | n_batch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |     256 |          pp5000 |       2053.78 ± 3.98 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |     512 |          pp5000 |       2117.14 ± 3.84 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |    1024 |          pp5000 |       2114.64 ± 2.72 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |    2048 |          pp5000 |       2113.36 ± 1.07 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |    4096 |          pp5000 |       2109.16 ± 1.64 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |    8192 |          pp5000 |       2108.36 ± 1.05 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |   10240 |          pp5000 |       2103.35 ± 2.17 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |     256 |          pp5000 |       1906.74 ± 2.66 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |     512 |          pp5000 |       1979.42 ± 3.43 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |    1024 |          pp5000 |       1981.27 ± 1.96 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |    2048 |          pp5000 |       1977.15 ± 3.91 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |    4096 |          pp5000 |       1976.45 ± 1.40 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |    8192 |          pp5000 |       1977.29 ± 1.33 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |   10240 |          pp5000 |       1974.94 ± 2.86 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |     256 |          pp5000 |       1995.60 ± 4.79 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |     512 |          pp5000 |       2118.83 ± 1.57 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |    1024 |          pp5000 |       2115.92 ± 3.24 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |    2048 |          pp5000 |       2113.88 ± 3.55 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |    4096 |          pp5000 |       2113.55 ± 4.67 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |    8192 |          pp5000 |       2113.17 ± 3.35 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |   10240 |          pp5000 |       2111.67 ± 2.69 |
```
Our desired `Qwen3.8-27B` model is pretty speedy at `2111 TPS` and is just as fast as the smaller `3.6` one


## Performance vs. batch-size @ 10000 token Prompt size

```text
$ ../build/bin/llama-bench  --model ./Qwen3.6-27B-UD-Q4_K_XL.gguf --model ./Qwen3.8-27B-UD-Q6_K_XL.gguf --model ./Qwen3.8-27B-UD-Q8_K_XL.gguf --batch-size 256,512,1024,2048,4096,8192,10240 --n-prompt 10000 --n-gen 0
ggml_cuda_init: found 2 CUDA devices (Total VRAM: 48214 MiB):
  Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes, VRAM: 24124 MiB
  Device 1: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes, VRAM: 24090 MiB
| model                          |       size |     params | backend    | ngl | n_batch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |     256 |         pp10000 |       2047.40 ± 2.92 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |     512 |         pp10000 |       2165.56 ± 3.37 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |    1024 |         pp10000 |       2154.68 ± 0.86 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |    2048 |         pp10000 |       2153.61 ± 2.14 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |    4096 |         pp10000 |       2147.45 ± 1.56 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |    8192 |         pp10000 |       2146.35 ± 1.75 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |   10240 |         pp10000 |       2144.15 ± 0.77 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |     256 |         pp10000 |       1891.03 ± 2.18 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |     512 |         pp10000 |       2009.78 ± 2.62 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |    1024 |         pp10000 |       2007.83 ± 1.70 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |    2048 |         pp10000 |       2009.81 ± 1.21 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |    4096 |         pp10000 |       2007.72 ± 3.72 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |    8192 |         pp10000 |       2008.18 ± 2.38 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |   10240 |         pp10000 |       2006.57 ± 1.87 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |     256 |         pp10000 |       1986.77 ± 1.54 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |     512 |         pp10000 |       2150.53 ± 2.22 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |    1024 |         pp10000 |       2149.31 ± 1.66 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |    2048 |         pp10000 |       2148.63 ± 2.95 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |    4096 |         pp10000 |       2147.58 ± 3.96 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |    8192 |         pp10000 |       2148.94 ± 1.70 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |   10240 |         pp10000 |       2145.86 ± 3.17 |
```
ditto qu'en haut (above)


## Performance vs. ubatch

```text
$ ../build/bin/llama-bench  --model ./Qwen3.6-27B-UD-Q4_K_XL.gguf --model ./Qwen3.8-27B-UD-Q6_K_XL.gguf --model ./Qwen3.8-27B-UD-Q8_K_XL.gguf --ubatch-size 256,512,1024,2048,4096,8192 --n-prompt 0  --n-gen 500,1000

| model                          |       size |     params | backend    | ngl | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |      256 |           tg500 |         42.12 ± 0.02 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |      256 |          tg1000 |         42.00 ± 0.08 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |      512 |           tg500 |         41.70 ± 0.07 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |      512 |          tg1000 |         41.51 ± 0.04 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |     1024 |           tg500 |         41.48 ± 0.01 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |     1024 |          tg1000 |         41.45 ± 0.02 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |     2048 |           tg500 |         41.47 ± 0.02 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |     2048 |          tg1000 |         41.44 ± 0.01 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |     4096 |           tg500 |         41.45 ± 0.02 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |     4096 |          tg1000 |         41.44 ± 0.02 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |     8192 |           tg500 |         41.44 ± 0.01 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |     8192 |          tg1000 |         41.42 ± 0.01 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |      256 |           tg500 |         30.76 ± 0.01 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |      256 |          tg1000 |         30.73 ± 0.01 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |      512 |           tg500 |         30.76 ± 0.01 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |      512 |          tg1000 |         30.74 ± 0.01 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |     1024 |           tg500 |         30.76 ± 0.01 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |     1024 |          tg1000 |         30.72 ± 0.00 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |     2048 |           tg500 |         30.75 ± 0.00 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |     2048 |          tg1000 |         30.72 ± 0.00 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |     4096 |           tg500 |         30.75 ± 0.01 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |     4096 |          tg1000 |         30.71 ± 0.01 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |     8192 |           tg500 |         30.76 ± 0.00 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |     8192 |          tg1000 |         30.72 ± 0.01 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |      256 |           tg500 |         26.48 ± 0.01 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |      256 |          tg1000 |         26.45 ± 0.00 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |      512 |           tg500 |         26.48 ± 0.00 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |      512 |          tg1000 |         26.47 ± 0.00 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |     1024 |           tg500 |         26.48 ± 0.00 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |     1024 |          tg1000 |         26.47 ± 0.00 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |     2048 |           tg500 |         26.49 ± 0.00 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |     2048 |          tg1000 |         26.47 ± 0.00 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |     4096 |           tg500 |         26.49 ± 0.01 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |     4096 |          tg1000 |         26.47 ± 0.00 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |     8192 |           tg500 |         26.49 ± 0.01 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |     8192 |          tg1000 |         26.48 ± 0.00 |
```
okay, so really no variance for `Qwen3.8-27B`, holding stead at the expected `26 TPS`. The model (Q4, Q6, Q8) quantization def makes 
a difference here



### Performance vs. Flash Attention

```text
$ ../build/bin/llama-bench  --model ./Qwen3.6-27B-UD-Q4_K_XL.gguf --model ./Qwen3.8-27B-UD-Q6_K_XL.gguf --model ./Qwen3.8-27B-UD-Q8_K_XL.gguf --n-depth 0,5000,10000,20000,50000 --flash-attn 0,1
| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |   0 |           pp512 |      1345.96 ± 17.24 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |   0 |           tg128 |         41.75 ± 0.04 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |   0 |   pp512 @ d5000 |      1184.20 ± 11.13 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |   0 |   tg128 @ d5000 |         39.97 ± 0.06 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |   0 |  pp512 @ d10000 |       1058.38 ± 8.51 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |   0 |  tg128 @ d10000 |         38.35 ± 0.03 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |   0 |  pp512 @ d20000 |        876.37 ± 6.54 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |   0 |  tg128 @ d20000 |         31.91 ± 0.03 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |   0 |  pp512 @ d50000 |        487.11 ± 2.24 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |   0 |  tg128 @ d50000 |         21.59 ± 0.01 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |   1 |           pp512 |      1324.53 ± 12.86 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |   1 |           tg128 |         41.99 ± 0.01 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |   1 |   pp512 @ d5000 |       1246.56 ± 9.29 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |   1 |   tg128 @ d5000 |         41.20 ± 0.02 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |   1 |  pp512 @ d10000 |       1182.61 ± 8.08 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |   1 |  tg128 @ d10000 |         40.67 ± 0.03 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |   1 |  pp512 @ d20000 |       1074.42 ± 8.87 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |   1 |  tg128 @ d20000 |         39.40 ± 0.06 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |   1 |  pp512 @ d50000 |        846.75 ± 6.50 |
| qwen35 27B Q4_K - Medium       |  16.39 GiB |    26.90 B | CUDA       |  -1 |   1 |  tg128 @ d50000 |         36.10 ± 0.04 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |   0 |           pp512 |      1251.08 ± 12.13 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |   0 |           tg128 |         30.71 ± 0.01 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |   0 |   pp512 @ d5000 |      1103.40 ± 11.50 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |   0 |   tg128 @ d5000 |         29.73 ± 0.01 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |   0 |  pp512 @ d10000 |        998.16 ± 7.75 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |   0 |  tg128 @ d10000 |         28.83 ± 0.02 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |   0 |  pp512 @ d20000 |        839.53 ± 5.88 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |   0 |  tg128 @ d20000 |         25.04 ± 0.01 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |   0 |  pp512 @ d50000 |        477.74 ± 2.43 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |   0 |  tg128 @ d50000 |         18.22 ± 0.01 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |   1 |           pp512 |      1262.74 ± 11.02 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |   1 |           tg128 |         30.86 ± 0.01 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |   1 |   pp512 @ d5000 |      1194.68 ± 10.22 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |   1 |   tg128 @ d5000 |         30.44 ± 0.02 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |   1 |  pp512 @ d10000 |       1137.02 ± 8.85 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |   1 |  tg128 @ d10000 |         30.18 ± 0.02 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |   1 |  pp512 @ d20000 |       1037.78 ± 9.45 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |   1 |  tg128 @ d20000 |         29.48 ± 0.03 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |   1 |  pp512 @ d50000 |        822.68 ± 7.48 |
| qwen35 27B Q4_K - Small        |  24.13 GiB |    27.32 B | CUDA       |  -1 |   1 |  tg128 @ d50000 |         27.63 ± 0.02 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |   0 |           pp512 |      1328.59 ± 17.32 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |   0 |           tg128 |         26.38 ± 0.00 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |   0 |   pp512 @ d5000 |      1166.39 ± 10.98 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |   0 |   tg128 @ d5000 |         25.66 ± 0.01 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |   0 |  pp512 @ d10000 |       1049.20 ± 9.15 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |   0 |  tg128 @ d10000 |         24.97 ± 0.02 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |   0 |  pp512 @ d20000 |        874.18 ± 6.21 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |   0 |  tg128 @ d20000 |         22.08 ± 0.01 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |   0 |  pp512 @ d50000 |        488.61 ± 2.57 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |   0 |  tg128 @ d50000 |         16.60 ± 0.01 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |   1 |           pp512 |      1343.35 ± 12.17 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |   1 |           tg128 |         26.50 ± 0.01 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |   1 |   pp512 @ d5000 |      1269.92 ± 11.51 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |   1 |   tg128 @ d5000 |         26.19 ± 0.01 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |   1 |  pp512 @ d10000 |       1206.37 ± 9.92 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |   1 |  tg128 @ d10000 |         26.00 ± 0.02 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |   1 |  pp512 @ d20000 |       1096.90 ± 9.09 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |   1 |  tg128 @ d20000 |         25.46 ± 0.03 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |   1 |  pp512 @ d50000 |        861.12 ± 4.97 |
| qwen35 27B Q4_K - Medium       |  29.29 GiB |    27.32 B | CUDA       |  -1 |   1 |  tg128 @ d50000 |         24.07 ± 0.03 |
```
Bref, we need to keep Flash Attention on, otherwise at big (50K KV cache) text-generation of 128 tokens can slow down to 16TPS!

### So what have we learned ..???
- Defaults for `batch-size/ubatch` are good enough
- Flash Attention is crucial, otherwise we might lose 10 TPS
- Expanding context size can slow decode TPS to from 26 &rarr; 22 TPS
- `Qwen3.8-27B` is plenty fast for prompt-processing 
- We can expect real-world performance between 20-26 TPS, which is acceptable for our misc. tasks
- More work is required to optimize the sampler settings for coherence, text-generation quality


## References

- Benchmark deep dive inspired by [ewams post ](https://ewams.net/?date=2026%2F04%2F26&view=How_to_Benchmark_AI_Performance_with_llama-bench_%28llama.cpp%29)on the matter
- `Usage: llama-bench` &rarr; https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md