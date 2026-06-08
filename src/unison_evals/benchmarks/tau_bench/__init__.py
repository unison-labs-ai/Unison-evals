"""τ-bench integration — Sierra + Anthropic's multi-turn workspace-orchestration
benchmark (retail + airline domains).

Mode A (baseline): the agent uses τ-bench's native typed function-call tools.
Mode B (Unison): the same agent model uses one `bash` tool over a virtual
`.md` filesystem, mutations dispatched through an md-overlay adapter.

Pinned upstream commit: 59a200c (vendored at unison-evals/vendor/tau-bench).
"""
