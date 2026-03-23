# Claude Code Prompt

You are in the `autoresearch-go` repository and should act as an autonomous research engineer.

Start by reading:

- `program.md`
- `README.md`
- `prepare.go`
- `train.go`

After that, run the autoresearch loop autonomously.

Constraints:

- Edit only `train.go`.
- Treat `prepare.go` as read-only.
- Do not modify evaluation logic.
- Do not install packages or add dependencies.
- Log every experiment in `results.tsv`.
- Keep the working branch dedicated to autoresearch.

Loop:

1. Determine the current best result from `results.tsv`.
2. Propose one concrete experiment in `train.go`.
3. Apply the change.
4. Run any quick verification needed.
5. Commit the experiment.
6. Execute:

   ```bash
   go run ./cmd/train > run.log 2>&1
   ```

7. Read `run.log`, extract `val_bpb` and `peak_heap_mb`, and append the result to `results.tsv`.
8. If the new `val_bpb` is lower, keep the commit.
9. If the new `val_bpb` is equal or worse, reset to the previous best commit.
10. Continue with the next idea without waiting for user approval.

Optimization target:

- Lower `val_bpb` under the fixed training budget.
- Favor coherent, reviewable changes.
- Avoid complexity unless it clearly improves the metric.
