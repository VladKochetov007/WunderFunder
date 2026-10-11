# H382 Kaggle delivery R2

R2 is the identified delivery repair for H382's three preregistered fold-0
fits. R1's immutable packet and startup-only failure are preserved in the
separate R1 directory and operator receipt.

The R2 packet is `H382_PACKET_R2.json`. Its kernel is
`vladkochetov07/wnn-kaggle-crossfold-h382-20261011-r2`, its private code bundle
dataset is `vladkochetov07/wnn-kaggle-crossfold-h382-bundle-20261011-r2`, and
the bundle is SHA-256
`639725151b75914aa759539e0de1fffd42d2e67baaab477e968bd46305ebe4c3` (355,804
bytes). The bootstrap is 26,724 bytes. R2 asks Kaggle for
`NvidiaTeslaT4` both in metadata and at push, and stops before unpacking input
or fitting unless `nvidia-smi` reports exactly two T4 devices.

R1's provider output showed the split and portable registry already present
from the verified bundle, while the worker also tried to symlink mounted copies
over them. R2 still checks each mounted copy against the asset manifest and
checks the existing bundle materialization by size and digest. The raw input
linking, source binding, recipes, membership, and exports remain unchanged.

Rebuild only before any provider upload with:

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=50% WNN_THREADS=1 \
WNN_MEM_MAX=800000000 WNN_RUNTIME_MAX=120s \
scripts/limited .venv/bin/python build_packet.py
```

The packet validator also checks the real completed V6 full-validation
prediction and sidecar. Provider upload is an operator action, not performed
by the packet builder. For a push, keep the short physical staging folder to
the two frozen upload files and pass `--accelerator NvidiaTeslaT4` explicitly.
