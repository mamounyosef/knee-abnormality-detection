# Prompt for the provisioning agent

Copy everything below the line into a fresh Claude Code session opened in
`D:\My Projects\knee_abnormality_detection`.

---

You are provisioning data for a Kaggle competition project across many Modal
workspaces. This is a long-running babysitting job: start work, wait, restart
what fails, report briefly. Do not change any modelling code.

## Goal

Every Modal workspace listed in `modal_setup_commands.txt` should end up holding
**4,407 preprocessed study shards (~137 GiB) on the `knee-shards` volume**, with
the ~247 GiB competition zip deleted afterwards to free space.

Workspace **3 (`modal`) is already complete** — leave it alone.
Workspace **1 (`modal`) keeps its zip permanently** as the archive
backup; do not delete that one. Its credits are exhausted, so it may fail to
preprocess — if so, skip it and note that in your report.

## The one command that does everything

```powershell
python modal_scripts/pipeline.py --launch-all --watch --batch 3 --auto-delete-zip
```

- `--batch 3` keeps only **3 workspaces working at a time**. This matters:
  Kaggle throttles many simultaneous downloads from one account, and we already
  hit HTTP 429s at higher concurrency. As each workspace finishes, the next one
  starts automatically.
- `--auto-delete-zip` removes the zip once a workspace reaches 4,407 shards
  (skipping session 1). Safe: shards keep every series and every slice.
- `--watch` refreshes the status table and rewrites
  `data/stats/pipeline_status.md`.

**Run it as a background process** (Claude Code's `run_in_background`), not in
the foreground — it runs for many hours. Then poll its output file and
`data/stats/pipeline_status.md` occasionally rather than blocking on it.

## What each workspace goes through

1. `modal_scripts/modal_download.py` — pulls the competition zip from Kaggle
   straight onto the `knee-data` volume (~1.5 h, resumable via HTTP Range).
2. `modal_scripts/modal_preprocess.py` — decodes it into one `.npz` per study on
   `knee-shards` (~14 min on 20 containers, ~137 GiB).
3. zip deleted.

`pipeline.py` picks the right stage per workspace automatically, so you never
need to call those two scripts yourself.

## Environment gotchas

- **Always set the workspace before any modal command**:
  `$env:MODAL_CONFIG_PATH = "$HOME\.modal_session<N>.toml"`.
  `modal_setup_commands.txt` maps session numbers to workspace names — read it
  first so you know the order. `pipeline.py` handles this internally.
- **TLS**: Avast intercepts HTTPS on this machine. Before any Python network
  call from PowerShell:
  `$env:REQUESTS_CA_BUNDLE="$HOME\.certs\ca-bundle.pem"; $env:SSL_CERT_FILE=$env:REQUESTS_CA_BUNDLE`
  (`pipeline.py` sets this for its own subprocesses.)
- `modal` is not on PATH; invoke it as `python -m modal`.

## Monitoring

- `python modal_scripts/pipeline.py` — one-shot status table, costs nothing
  (`modal volume ls` is a control-plane call, no containers).
- `data/stats/pipeline_status.md` — same table, rewritten each refresh.

## Failure handling

- A workspace stuck at the same shard count for >30 min: re-run
  `python modal_scripts/pipeline.py --launch <N>`. Both stages resume, so
  nothing already downloaded or preprocessed is repeated.
- Out of credit (403/402, or jobs dying instantly): skip that workspace, record
  it, move on.
- HTTP 429 from Kaggle: lower to `--batch 2` and continue.
- Do **not** raise `--batch` above 3.

## Reporting

Keep every report to the user **very short** — a few lines. Report:

- when the first batch of 3 is under way,
- each time a workspace reaches READY (with the running count, e.g. "5/18 ready"),
- any workspace you skipped and why,
- a final one-line summary.

Do not paste long logs. Do not explain what you are about to do at length. The
status table is the report.
