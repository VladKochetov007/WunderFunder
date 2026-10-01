# Offline verification runtime

Prepare isolated Python3.11 dependencies from the root published requirements;
do not alter the research virtual environment. Freeze installed distribution
versions and file hashes after the durable bootstrap exits. Cached
python:3.11-slim-bookworm is the base named by submission/Dockerfile.build;
bind the installed manylinux dependencies when testing with network disabled.
Record the image ID rather than treating its mutable tag as an identity.

This is a local scorer-compatible image/runtime test, not execution on the
competition portal. It establishes offline Python/ABI/dependency compatibility;
public score association and platform timing remain separate evidence.

Bootstrap uses one CPU,2GiB hard memory and wheel-only installs. Verification
containers must use one CPU,<=16GiB RAM,no network,read-only source/data mounts
and an isolated output/temp directory. Root coordinates aggregate CPU/RAM with
GPU research jobs. No upload or external model/data state is involved.
