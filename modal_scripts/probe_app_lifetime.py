r"""probe_app_lifetime.py - can this workspace run an app for longer than ten minutes?

WHY. Session 23 killed every one of its download apps at almost exactly 10 minutes with
"Received a cancellation signal", and the container was doing nothing unusual: it connected
to Kaggle in 0.4 s and was streaming. Two separate explanations fit that: the workspace caps
app lifetime, or something about the downloader provoked it.

The distinction decides whether session 23 is worth any more effort. A workspace that stops
apps at 10 minutes cannot host a 9 hour training run, so no amount of fixing the DOWNLOAD
would make it usable for its actual purpose.

This does nothing but sleep and print a heartbeat, on CPU, so it costs about a cent and
shares no code path with the downloader. If it reaches 15 minutes, the cap is not real and
the downloader is provoking the cancellation. If it dies near 10, the workspace is the cap.

    $env:MODAL_CONFIG_PATH = "$HOME\.modal_session23.toml"
    python -m modal run modal_scripts/probe_app_lifetime.py
"""

import time

import modal

app = modal.App("lifetime-probe")
image = modal.Image.debian_slim(python_version="3.11")

MINUTES = 15


@app.function(image=image, timeout=60 * 60)
def sit_still():
    t0 = time.time()
    print(f"[probe] start, intending to sit for {MINUTES} minutes", flush=True)
    while True:
        el = time.time() - t0
        if el >= MINUTES * 60:
            break
        print(f"[probe] +{el / 60:5.2f} min alive", flush=True)
        time.sleep(30)
    print(f"[probe] SURVIVED {MINUTES} minutes - this workspace does NOT cap app lifetime",
          flush=True)
    return el


@app.local_entrypoint()
def main():
    el = sit_still.remote()
    print(f"probe returned after {el / 60:.2f} min")
