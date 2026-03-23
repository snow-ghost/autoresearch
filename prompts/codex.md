# Codex Prompt

You are running inside the `autoresearch-go` repository.

Read these files first:

- `program.md`
- `README.md`
- `prepare.go`
- `train.go`

Then continue the autonomous research loop without asking for confirmation between experiments.

Rules:

- Work on the current autoresearch branch.
- Modify only `train.go`.
- Do not modify `prepare.go`.
- Do not change the evaluation harness.
- Do not add dependencies.
- Use `results.tsv` as the experiment log.
- Keep `results.tsv` and `run.log` out of normal experiment commits unless explicitly requested.

Process:

1. Check the current branch and current best commit.
2. Inspect `results.tsv` and identify the current best `val_bpb`.
3. Make one experimental change in `train.go`.
4. Run tests if needed.
5. Commit the change with a short message.
6. Run:

   ```bash
   go run ./cmd/train > run.log 2>&1
   ```

7. Extract metrics from `run.log`.
8. Append a row to `results.tsv`.
9. If `val_bpb` improved, keep the commit and continue from there.
10. If `val_bpb` did not improve, reset back to the previous best commit and continue with a new idea.

Goal:

- Minimize `val_bpb`.
- Prefer simple changes over complex ones when the gains are similar.
- Continue autonomously until explicitly stopped.
