# Working together

Two maintainers: Mike Lu (3apart) and Joel. Anyone else needs a signed NDA with 3apart before they're added.

## How changes get in

- `main` is protected: every change is a pull request, approved by the other maintainer.
- **Results** (new files under `results/` and the regenerated `results/SUMMARY.md`): approve once CI is green. Never edit someone else's result file; add a new run instead.
- **New engines or benchmarks** (`engines/`, `bench/`): fine to add, as long as they follow PROTOCOL.md and the engine contract.
- **Protocol changes** (anything in `src/`, the physics forces, run counts, tolerances, PROTOCOL.md): both maintainers approve, `schema` goes up, and PROTOCOL.md gets a dated note. Old results stay and are compared only with their own schema.
- Keep the d3-force-3d version pinned. Upgrading it is a protocol change.

## Result files

`results/<who>/<machine>/<UTC timestamp>-<kind>.json`, written by the runners. Commit them as they come out; `npm run compare` rebuilds the summary. Each file carries the machine, versions, the git commit it ran on, and every run, so numbers can always be traced.

## Confidentiality

- This repo is the patent-pending 3apart method. Keep it private, don't copy it into other repos or public gists, and don't paste it into tools that keep or train on what you give them.
- Synthetic data only. Real data from partners (for example Vatic's market data) never goes in this repo; a benchmark on real data gets its own private repo with its own access rules.
- Contributions to the placement method itself can bear on inventorship. If you're changing the method rather than measuring it, flag it in the pull request so 3apart's patent counsel can look at it.
