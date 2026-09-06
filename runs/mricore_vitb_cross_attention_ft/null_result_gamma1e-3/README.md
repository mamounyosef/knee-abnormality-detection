# Null result: gamma_init 1e-3 (6 Sep)

Kept as evidence, not as a run to build on. The gate has since been **deleted outright**:
the block is now `out = x + attn(norm(x))`, with no scale parameter at all.

## What happened

Started from `mricore_vitb_relabel` (val 0.8970) and spliced in three CrossSliceBlocks
gated by a per-channel `gamma` initialised to 1e-3.

| step | val macro AUROC |
|---|---|
| init | **0.8970** |
| 150 | 0.8946 |
| 450 | 0.8952  (peak) |
| 900 | 0.8941 |
| 1500 (session ended) | 0.8936 |

Gate, block 3:

| step | \|gamma\| mean | channels open |
|---|---|---|
| 0 | 0.00091 | 296 / 768 |
| 150 | 0.00061 | 122 / 768 |
| 300 | 0.00047 | 82 / 768 |
| 450 | 0.00040 | 52 / 768 |

## Why it proves nothing about cross-slice attention

The gate was a trap, not a measurement. The gradient reaching the branch's OWN weights is
scaled by gamma, so at 1e-3 the attention trained ~1000x too slowly, while gamma's own
gradient was evaluated on a branch that was still random noise. So the gate closed, which
kept the branch untrained, which kept the gate closed. Self-reinforcing.

The run never tested whether cross-slice information helps. It tested whether a branch
throttled to 0.1% strength can bootstrap itself out of that throttle, and the answer is no.

Weight decay is NOT the cause: at lr 6e-5 and wd 0.05 the decoupled decay is 3e-6 of the
value per step, about 0.2% over 600 steps.

## Also found here

Every gamma value printed to stdout was LOST from `train_log.csv`. `CSVLogger` locked its
header on row 1, before any gate row existed, and `DictWriter(extrasaction="ignore")` then
dropped the columns silently for the whole run. The numbers above were recovered from
terminal scrollback. Fixed in `shared_code.py`: the logger now widens the file mid-run.
