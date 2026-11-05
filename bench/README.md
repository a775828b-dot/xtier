# bench

Acceptance against the baselines of the current production pipelines: speed, PPL, verbatim-copy test, long-context recall.

The console performance panel reuses the Agent and copy checks and provides a unified HTTP runner for decode, concurrency, context, and cache tests. See [performance panel](../docs/performance-panel.md) for commands, metric definitions, and limits. The Agent script supports both the former adapter/ layout and the current xtier/ layout; results from these different source tasks are not interchangeable baselines.
