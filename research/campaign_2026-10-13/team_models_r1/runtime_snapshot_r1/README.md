# Immutable replay snapshot

Live originals remain on disk and are excluded from Git. These immutable snapshots preserve current useful evidence while four healthy replays continue. Completed scores are indexed separately in the campaign score README.

- [Manifest](MANIFEST.json): original source paths, snapshot paths, byte counts, SHA256 and capture time.
- [REPLAY_CONTROLLER.log](REPLAY_CONTROLLER.log): complete-line snapshot of callback execution/progress. It records observed status at capture; partial progress does not certify completion.
- [scores](scores/README.md): captured runtime evidence for this group; see the manifest for source and hashes.
