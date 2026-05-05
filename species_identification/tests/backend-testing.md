# Backend setup for `eval_e2e.py`

The harness is the same; the backends differ. The goal is to get a fair
comparison: same model weights, same sampling, same prompt — so any
divergence is attributable to the runtime, not config drift.

## 1. Pick the model file once

You already have `smollm2-360m-q8_0.gguf` for the Uno Q. Use the same GGUF
on the laptop. Don't use Ollama's stock `smollm2:360m` for benchmarking —
its quant level may differ from yours and you'll be comparing apples to
oranges.

```
roboranger/
  models/
    smollm2-360m-q8_0.gguf      <-- single source of truth
  runs/                          <-- saved eval outputs go here
```

## 2. Laptop / Ollama

Ollama can load arbitrary GGUFs through a `Modelfile`:

```
# models/Modelfile
FROM ./smollm2-360m-q8_0.gguf
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>"""
PARAMETER stop "<|im_end|>"
```

Register it once:

```bash
ollama create smollm2-q8 -f models/Modelfile
ollama list   # confirm 'smollm2-q8' is there
```

Then run the harness:

```bash
python eval_e2e.py --backend ollama --model smollm2-q8 \
                   --save runs/laptop.json
```

Note: `eval_e2e.py` already pins temperature, top_p, top_k, and seed via
the `SAMPLING` dict, so Ollama's defaults won't leak in.

## 3. Uno Q / llama.cpp

Build llama.cpp on the Uno Q (Cortex-A53 aarch64) — you've already done
this for benchmarking. Start the server with the same GGUF:

```bash
./llama-server \
  -m /path/to/smollm2-360m-q8_0.gguf \
  --host 0.0.0.0 --port 8080 \
  --ctx-size 2048 \
  --threads 4 \
  --chat-template chatml
```

Notes:
- `--chat-template chatml` matches what `prompt_builder.py` produces. If you
  feed `messages` to the OpenAI-compatible endpoint, llama.cpp applies the
  template itself — that's what `LlamaCppBackend` does.
- `--ctx-size 2048` is plenty: blurb (~800 chars) + 3 chunks (400 each) +
  system prompt is well under 1k tokens.
- `--threads 4` matches the QRB2210's A53 cores. Tune if you see CPU idle.

From the laptop, point the harness at the Uno Q:

```bash
python eval_e2e.py --backend llama-cpp \
                   --host http://uno-q.local:8080 \
                   --save runs/unoq.json
```

(Replace `uno-q.local` with the device's IP if mDNS doesn't resolve on your
network.)

## 4. Compare the two runs

```bash
python diff_runs.py runs/laptop.json runs/unoq.json
```

What to look for:

- **Pass/fail divergence** (`<-- DIVERGE` flag): a case the laptop passes
  but the Uno Q fails, or vice versa. Highest-priority signal — the same
  prompt is producing different behavior on the two runtimes.
- **Low keyword overlap** (`<-- low overlap` flag, < 0.4 Jaccard): the
  answers are saying different things even if both happen to pass the
  heuristics. Worth reading both responses to see why.
- **Latency**: expect Uno Q to be 5–20× slower than the laptop. If it's
  worse than that, suspect thread count or thermal throttling on the QRB2210.

## 5. Iteration loop

Recommended workflow:

1. Make a change to `prompt_builder.py` or the prototypes in `intent.py`.
2. Run `pytest test_units.py` — fast regression check.
3. Run `eval_intent.py` — confirms intent classification still works.
4. Run `eval_e2e.py --backend ollama --save runs/laptop.json` — fast
   end-to-end on the laptop.
5. Read the responses. Iterate on 1–4 until laptop output looks right.
6. Only then run on the Uno Q and `diff_runs.py`. This is the slow step,
   so you don't want to do it on every change.

## 6. Things that will bite you

- **Ollama defaults.** If you forget to specify `--model smollm2-q8` and
  fall back to `smollm2:360m` from the registry, you're benchmarking a
  different quantization. The harness prints the model name in saved runs
  — check it.
- **Different chat templates.** `prompt_builder.py` emits ChatML. Both
  backends must apply ChatML on top of `messages`. If llama.cpp is
  configured with a different template, the model will see malformed input
  and produce garbage. Confirm by reading the server logs once.
- **Seeds aren't always honored.** Ollama respects `seed`. llama.cpp
  respects it for the sampler but kv-cache reuse can introduce
  non-determinism across runs. Don't expect byte-identical outputs even
  with the same seed — focus on pass/fail and content overlap, not exact
  string match.
- **First run on Uno Q is slow.** First inference loads the model into
  memory. The latency numbers in the saved run include this if the server
  was just started. Run once to warm up before the eval that you save.