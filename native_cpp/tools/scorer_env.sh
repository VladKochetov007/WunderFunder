#!/usr/bin/env bash
# Build a runnable copy of the scorer image (python:3.11-slim-bookworm, glibc 2.36) without docker,
# chroot or user namespaces: the image layers are pulled from Docker Hub, unpacked, and its Python is
# started through the image's own dynamic loader with the image libraries.
#   native_cpp/tools/scorer_env.sh DIR            -> DIR/run.sh script.py args...
# Python deps of the scorer (numpy, pyarrow, onnxruntime pins) go to DIR/site via uv; set PY311 to a
# local Python 3.11 so uv resolves cp311 wheels.
set -euo pipefail
dir=$(realpath -m "${1:?target directory}")
mkdir -p "$dir/layers" "$dir/rootfs"
cd "$dir"
tok=$(curl -s "https://auth.docker.io/token?service=registry.docker.io&scope=repository:library/python:pull" |
  python3 -c 'import json, sys; print(json.load(sys.stdin)["token"])')
auth="Authorization: Bearer $tok"
curl -s -H "$auth" -H "Accept: application/vnd.oci.image.index.v1+json" \
  -H "Accept: application/vnd.docker.distribution.manifest.list.v2+json" \
  https://registry-1.docker.io/v2/library/python/manifests/3.11-slim-bookworm -o index.json
digest=$(python3 -c 'import json; m = json.load(open("index.json")); print([x["digest"] for x in m["manifests"] if x["platform"]["architecture"] == "amd64" and x["platform"]["os"] == "linux"][0])')
curl -s -H "$auth" -H "Accept: application/vnd.oci.image.manifest.v1+json" \
  -H "Accept: application/vnd.docker.distribution.manifest.v2+json" \
  "https://registry-1.docker.io/v2/library/python/manifests/$digest" -o manifest.json
layers=$(python3 -c 'import json; print(" ".join(l["digest"] for l in json.load(open("manifest.json"))["layers"]))')
for layer in $layers; do
  curl -sL -H "$auth" "https://registry-1.docker.io/v2/library/python/blobs/$layer" -o "layers/${layer#sha256:}.tgz"
  tar -xzf "layers/${layer#sha256:}.tgz" -C rootfs --exclude='dev/*' 2>/dev/null || true
done
uv pip install -q --target "$dir/site" --python "${PY311:-python3.11}" numpy==2.2.6 pyarrow==19.0.1 onnxruntime==1.23.2
cat > run.sh <<RUN
#!/usr/bin/env bash
R=$dir/rootfs
export PYTHONHOME=\$R/usr/local PYTHONPATH=$dir/site\${PYTHONPATH:+:\$PYTHONPATH}
exec \$R/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2 --library-path \$R/lib/x86_64-linux-gnu:\$R/usr/lib/x86_64-linux-gnu:\$R/usr/local/lib \$R/usr/local/bin/python3.11 "\$@"
RUN
chmod +x run.sh
./run.sh -c 'import platform, sys; print("scorer env ready:", sys.version.split()[0], platform.libc_ver())'
