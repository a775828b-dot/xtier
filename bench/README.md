# bench

Acceptance against the baselines of the current production pipelines: speed, PPL, verbatim-copy test, long-context recall.

`agent_bench.py` runs from a checkout of the former adapter/ layout or of the xtier/ layout; the tasks of the two are
not interchangeable baselines.

`abuse_test.py`: what clients do to a running server that it must survive (aborted streams in the prefill and the
decode, requests to refuse, three at once), each group followed by a small request that must be answered.
