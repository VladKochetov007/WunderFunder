# ADR0012: Independently implement shared teammate methods

2026-10-13 successor: [ADR0035](0035-official-team-code-and-weight-sharing.md). Independent-reimplementation restriction superseded for the official team.

Status: Accepted, 2026-10-07.

The user permits reading teammates' code and sharing ideas and implementation details, but requires independently written code. They report that identical teammate code can be banned. This instruction supersedes the earlier request to directly run Maxim's implementation.

Use inspected source as methodological evidence. Describe its mathematical recipe, data treatment, architecture and execution details, then implement the method using our own project conventions, libraries and builds. Do not copy source, native binaries or packaging code, or disguise copying with renamed variables and formatting. Independently train our weights as required by ADR0004. Delegated actors receive the same restriction.

Record reference sources and own implementation ancestry separately. Comparisons must declare methodological differences rather than claiming exact replication from a loosely related continuation. Existing frozen reference snapshots remain read-only evidence; preserving them does not authorize including them in runnable code or submission archives.

Current submission work uses our existing Rust streaming engine and our independently trained C1 checkpoints. Offline ZIP correctness, quantization fidelity and runtime remain separate delivery gates. No public improvement is inferred from local scores alone.
