# OpenCode Prompt

You are operating in the `autoresearch-go` repository.

Read the repository context first:

- `program.md`
- `README.md`
- `prepare.go`
- `train.go`

Then start autonomous experimentation.

Hard constraints:

- Only modify `train.go`.
- Do not modify `prepare.go`.
- Do not alter the evaluation harness.
- Do not add new dependencies.
- Record each run in `results.tsv`.

Experiment loop:

1. Read `results.tsv` and find the current best experiment.
2. Change `train.go` with exactly one clear experimental idea.
3. Commit the change.
4. Run:

   ```bash
   go run ./cmd/train > run.log 2>&1
   ```

5. Parse the output from `run.log`.
6. Append the result to `results.tsv`.
7. If the experiment improved `val_bpb`, keep the commit.
8. If it regressed or failed, reset to the previous best commit.
9. Continue autonomously until stopped.

Success criteria:

- Lower `val_bpb`.
- Keep the codebase simple.
- Prefer stable, incremental progress over noisy changes.
