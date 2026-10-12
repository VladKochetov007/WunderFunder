# Treatment kernel R4 mount-discovery successor

R4 is a startup-compatibility successor to the sealed R22 packet. It keeps the
same kernel ID, pair plan, private datasets, source bundle, model recipes,
profile thresholds, root-release checks, worker specifications, checkpoint
policy, and output names. R4 changes only how the four declared Kaggle dataset
payload roots are resolved.

The resolver accepts the known flat input/slug layout and nested
input/datasets/owner/slug/version/payload layout (also direct owner/slug roots).
It first anchors on the exact owner and dataset slug, then searches payload
directories with a depth-8 and 8,192-directory bound. A candidate must contain
all declared files at their expected sizes. Exactly one candidate is required;
the resolver never chooses the first match. If several size candidates exist,
it checks declared content hashes to disambiguate. The large raw train/valid
parquet hashes are deliberately deferred to the unchanged frozen trainer; if
their dataset has multiple plausible payload roots, R4 refuses the ambiguity
instead of hashing many tens of gigabytes during mount discovery.

After discovery, the existing external-input preflight still checks the exact
source/asset inventories and hashes every non-parquet mounted input. The frozen
trainer remains responsible for complete raw-parquet byte verification before
fitting. The discovery receipt records dataset ID, selected relative mount,
layout, candidate count, and any disambiguation hashes in results_preflight.json.

offline_fixture_r4.py exercises all four actual dataset IDs in both flat and
nested version/payload layouts using symlinks to the existing staged files.
It includes same-leaf-name wrong-owner decoys, content-based disambiguation and
an identical-duplicate refusal check. The nested case also runs the existing
external-input consumer: all 18 declared observations pass, with 16 non-parquet
inputs SHA-verified and only raw train/valid deferred to the frozen trainer.
Its profile guard fixture copies the
actual kaggle_gpu_runner terminal finished record shape and actual
PROFILE_UPDATE_COUNTS, GRAD_STATS, and PROFILE trainer markers from the prior
R3 T4 profile. The provider UUID is replaced with a fixture sentinel. That
fixture validates parser/guard compatibility only; it is not an R22
treatment-device profile or a new resource qualification. The abbreviated
heartbeat record is explicitly rejected by the terminal profile guard.

Bounded offline validation:

    WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=50% WNN_THREADS=1
    WNN_MEM_MAX=700000000 WNN_RUNTIME_MAX=120s
    /home/vlad/development/wunder/wnn_connectome_starterpack/scripts/limited
    /usr/bin/python3 offline_fixture_r4.py

R4 is prepared but unuploaded. Root release, independent packet review,
fresh quota/resource admission, dataset READY readback, and an actual R22 T4
profile remain external gates. No provider action or model work was run while
preparing this successor.
