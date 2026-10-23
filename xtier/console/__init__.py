"""XTier console: a web page to launch one XTier model server at a time and follow it.

`python -m xtier.console` serves it (docs/console.md). core.Console holds the logic (the models from
configs/*/console.py, options, start / stop, loading phases and decode telemetry from the server log), ui the page,
so a host application can embed the card with its own routes and texts. Standard library only: it runs without
SGLang or torch.
"""
