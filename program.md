# autoresearch-go

This repository is a Go 1.26 adaptation of [`karpathy/autoresearch`](https://github.com/karpathy/autoresearch).

The idea is the same: the human edits `program.md`, the agent edits the training code, and the system keeps running fixed-budget experiments autonomously.

## Setup

To set up a new run, work with the human to:

1. Agree on a run tag based on today's date, for example `mar22`.
2. Create a branch named `autoresearch/<tag>` from the current main branch.
3. Read the in-scope files:
   - `README.md`
   - `prepare.go`
   - `train.go`
4. Verify the dataset cache exists under `~/.cache/autoresearch-go`. If not, run:

   ```bash
   go run ./cmd/prepare
   ```

5. Initialize `results.tsv` with this header:

   ```tsv
   commit	val_bpb	peak_heap_mb	status	description
   ```

6. Confirm setup and start the experiment loop.

## Experimentation

Each experiment is a single CPU training run launched as:

```bash
go run ./cmd/train
```

The default time budget is fixed at 5 minutes. The evaluation metric is `val_bpb`, where lower is better.

What you CAN do:

- Modify `train.go`.
- Change model architecture, hidden size, embedding size, learning rate, batch size, optimizer settings, and training logic.

What you CANNOT do:

- Modify `prepare.go`.
- Modify the evaluation harness.
- Add external dependencies just to make a single experiment work.

The goal is simple: lower `val_bpb` while keeping the code coherent. Simpler wins matter. A tiny improvement with ugly complexity is usually not worth keeping.

The first run must always establish the baseline with the default `train.go`.

## Output format

At the end of a run the script prints a summary like this:

```text
---
val_bpb:          3.812345
training_seconds: 300.0
total_seconds:    301.2
peak_heap_mb:     19.4
total_tokens_k:   256.0
num_steps:        4000
num_params_k:     74.5
context_len:      16
hidden_dim:       96
```

To extract the key metrics:

```bash
grep "^val_bpb:\|^peak_heap_mb:" run.log
```

## Logging results

After each experiment, append a row to `results.tsv`:

```tsv
commit	val_bpb	peak_heap_mb	status	description
abc1234	3.812345	19.4	keep	baseline
def5678	3.790112	20.1	keep	wider hidden layer
9876fed	3.900000	19.1	discard	lower context length
```

Rules:

1. Use the short git commit hash.
2. Use `keep`, `discard`, or `crash`.
3. For crashes, use `0.000000` and `0.0`.
4. Keep the description short and concrete.

Do not commit `results.tsv`.

## The Loop

Loop forever:

1. Check current branch and commit.
2. Change `train.go` with one experimental idea.
3. Commit the change.
4. Run:

   ```bash
   go run ./cmd/train > run.log 2>&1
   ```

5. Extract results:

   ```bash
   grep "^val_bpb:\|^peak_heap_mb:" run.log
   ```

6. If grep returns nothing, inspect:

   ```bash
   tail -n 50 run.log
   ```

7. Record the outcome in `results.tsv`.
8. If `val_bpb` improved, keep the commit and continue from there.
9. If `val_bpb` stayed flat or got worse, reset back to the previous good commit.

## Failures

- If a run exceeds 10 minutes, kill it and treat it as a failure.
- If the crash is a trivial bug in `train.go`, fix it and retry once.
- If the idea is fundamentally bad, log `crash`, revert, and move on.

## Autonomy

Once the loop starts, do not pause to ask the human whether to continue. Keep running experiments until the human explicitly stops you.
