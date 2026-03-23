# autoresearch-go

`autoresearch-go` is a Go 1.26 adaptation of [karpathy/autoresearch](https://github.com/karpathy/autoresearch): a tiny repository where an agent iterates on a single training file, runs a fixed-budget experiment, keeps improvements, and discards regressions.

Instead of a GPU-first PyTorch stack, this version uses a small byte-level language model written in pure Go and trained on CPU. The point is the same as upstream:

- the repository stays intentionally small;
- the agent only edits one file;
- the runtime and evaluation stay fixed;
- experiments are compared by `val_bpb` (validation bits per byte, lower is better).

## What matters

- `prepare.go` contains the fixed runtime: dataset preparation, byte tokenizer, batch sampling, and evaluation helpers.
- `train.go` contains the model, optimizer, and training loop. This is the single file an agent should modify.
- `program.md` contains the autonomous-research operating instructions for the agent.

The commands are split into `cmd/prepare` and `cmd/train`, but the research-critical code still lives in the three files above.

## Quick start

Requirements:

- Go 1.26+
- network access for the initial corpus download

Prepare the dataset cache:

```bash
go run ./cmd/prepare
```

Run a single experiment:

```bash
go run ./cmd/train
```

By default:

- the corpus is downloaded to `~/.cache/autoresearch-go`;
- the training time budget is fixed at 5 minutes;
- evaluation uses `val_bpb` on a held-out validation split.

For smoke tests and local iteration you can temporarily override:

```bash
AUTORESEARCH_TIME_BUDGET=10s AUTORESEARCH_EVAL_BYTES=1024 go run ./cmd/train
```

## Project structure

```text
prepare.go          fixed runtime utilities, prep, evaluation
train.go            model + optimizer + training loop (agent edits this)
program.md          autonomous-research instructions
cmd/prepare/main.go prepare command
cmd/train/main.go   train command
```

## Design choices

- Pure Go, no external ML dependencies.
- Byte-level tokenizer with a synthetic BOS token, so `bpb` stays natural and vocabulary-independent.
- Small MLP language model so experiments remain real, deterministic, and easy to mutate from a single file.
- Fixed runtime helpers in `prepare.go`, mirroring the separation used by the upstream Python project.

## Notes

- The default corpus source is Tiny Shakespeare from Karpathy's `char-rnn` repository.
- Peak memory is reported as Go heap usage (`peak_heap_mb`), not VRAM.
- This is intentionally a compact experimentation scaffold, not a production training system.

## Using Autonomous Agents

This repository is designed to work well with terminal-based coding agents such as Codex, Claude Code, and OpenCode.

Recommended setup:

- start the agent in the repository root;
- let it read `program.md`, `README.md`, `prepare.go`, and `train.go`;
- run it on a dedicated branch such as `autoresearch/<tag>`;
- let it modify only `train.go`;
- let it log experiment outcomes in `results.tsv`;
- keep the session running in `tmux` or `screen` if you want long unattended runs.

Typical workflow:

1. Prepare the cache once:

   ```bash
   go run ./cmd/prepare
   ```

2. Start your agent in this repository.
3. Give it a prompt that tells it to read `program.md` and continue autonomously.
4. Let it repeat this loop:
   - edit `train.go`;
   - commit the change;
   - run `go run ./cmd/train > run.log 2>&1`;
   - extract `val_bpb` and `peak_heap_mb` from `run.log`;
   - append the result to `results.tsv`;
   - keep the commit only if `val_bpb` improved.

Prompt templates are included in:

- `prompts/codex.md`
- `prompts/claude-code.md`
- `prompts/opencode.md`

All prompt files are written in English and can be copy-pasted directly into the corresponding agent.
