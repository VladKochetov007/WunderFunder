from __future__ import annotations

import ast
import difflib
import hashlib
import json
from pathlib import Path

from map_support import sha256

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def replace_once(source: str, original: str, proposed: str) -> str:
    """Reject stale or ambiguous trainer insertion contexts."""
    if source.count(original) != 1:
        raise ValueError(f"ambiguous or absent integration context: {original[:80]}")
    return source.replace(original, proposed)


def proposed_source() -> tuple[str, str]:
    """Build a trainer preview in memory without mutating active controls."""
    original = (ROOT / "scripts/train_rnn.py").read_text()
    proposed = replace_once(
        original,
        "from wnn.splits import dev_halves, file_sha256, load_campaign_split, valid_row_groups\n",
        "from wnn.splits import dev_halves, file_sha256, load_campaign_split, valid_row_groups\n"
        "from wnn.spread_aux import PackedSpreadCache, structural_loss, validate_spread_cache\n",
    )
    proposed = replace_once(
        proposed,
        "                 quote_cache_root=None):\n",
        "                 quote_cache_root=None, spread_cache_root=None):\n",
    )
    proposed = replace_once(
        proposed,
        "        self._quote = None\n",
        "        self._quote = None\n"
        "        self.spread_cache_root = Path(spread_cache_root) if spread_cache_root else None\n"
        "        self._spread = None\n",
    )
    proposed = replace_once(
        proposed,
        "            return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(scored), torch.from_numpy(p)\n"
        "        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(scored)\n",
        "            result = (torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(scored), torch.from_numpy(p))\n"
        "        else:\n"
        "            result = (torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(scored))\n"
        "        if self.spread_cache_root is not None:\n"
        "            if self._spread is None:\n"
        "                self._spread = PackedSpreadCache(self.spread_cache_root, self.split)\n"
        "            result += (torch.from_numpy(self._spread.read(self.indices[i])),)\n"
        "        return result\n",
    )
    proposed = replace_once(
        proposed,
        '    ap.add_argument("--aux-struct", type=float, default=0.0,\n',
        '    ap.add_argument("--spread-aux-mode", choices=["original", "proxy_known", "current_known"], default="original")\n'
        '    ap.add_argument("--spread-cache-root", default="", help="supervision-only fold0 known-code cache")\n'
        '    ap.add_argument("--aux-struct", type=float, default=0.0,\n',
    )
    proposed = replace_once(
        proposed,
        "    train_ds = Sequences(train_split, train_idx, teachers, prop_cache.get(train_split),\n",
        "    spread_provenance = None\n"
        '    if args.spread_aux_mode != "original":\n'
        "        if (campaign_split is None or campaign_split.fold != 0 or args.aux_struct <= 0\n"
        '                or not args.spread_cache_root or args.train_source != "train+valid"\n'
        '                or args.init != "c20260929_since_s3_f0@ep3" or args.seed != 3\n'
        '                or args.epochs != 1 or args.checkpoint_policy != "fixed"\n'
        "                or args.checkpoint_epoch != 0 or not args.no_eval_during_fit):\n"
        '            raise ValueError("known-code spread screen requires fixed one-epoch clean fold0 continuation from seed3 ep3")\n'
        "        spread_root = Path(args.spread_cache_root)\n"
        "        if not spread_root.is_absolute():\n"
        "            spread_root = ROOT / spread_root\n"
        "        spread_root = spread_root.resolve()\n"
        "        spread_provenance = validate_spread_cache(spread_root, campaign_split.sha256, train_idx,\n"
        "                                                   extra_valid, campaign_split.evaluation_groups)\n"
        '        if json.loads((spread_root / "metadata.json").read_text())["N_SP_AUX"] != N_SP_AUX:\n'
        '            raise ValueError("spread cache head dimension differs")\n'
        '        spread_provenance["mode"] = args.spread_aux_mode\n'
        '        spread_provenance["source_sha256"] = file_sha256(ROOT / "src/wnn/spread_aux.py")\n'
        "        args.spread_cache_root = str(spread_root)\n"
        "    elif args.spread_cache_root:\n"
        '        raise ValueError("spread cache supplied without a matched known-code mode")\n'
        "    train_ds = Sequences(train_split, train_idx, teachers, prop_cache.get(train_split),\n",
    )
    proposed = replace_once(
        proposed,
        "    train_ds = Sequences(train_split, train_idx, teachers, prop_cache.get(train_split),\n"
        "                         quote_path=args.aux_quote > 0, quote_cache_root=args.quote_cache_root)\n",
        "    train_ds = Sequences(train_split, train_idx, teachers, prop_cache.get(train_split),\n"
        "                         quote_path=args.aux_quote > 0, quote_cache_root=args.quote_cache_root,\n"
        "                         spread_cache_root=args.spread_cache_root)\n",
    )
    proposed = replace_once(
        proposed,
        '        valid_ds = Sequences("valid", extra_valid, teachers, prop_cache.get("valid"),\n'
        "                             quote_path=args.aux_quote > 0, quote_cache_root=args.quote_cache_root)\n",
        '        valid_ds = Sequences("valid", extra_valid, teachers, prop_cache.get("valid"),\n'
        "                             quote_path=args.aux_quote > 0, quote_cache_root=args.quote_cache_root,\n"
        "                             spread_cache_root=args.spread_cache_root)\n",
    )
    proposed = replace_once(
        proposed,
        "    if quote_cache_provenance is not None:\n",
        "    if spread_provenance is not None:\n"
        '        provenance["spread_aux"] = spread_provenance\n'
        "    if quote_cache_provenance is not None:\n",
    )
    original_batch = """            prop_cached = None
            if args.pin_memory:
                (x, y, scored_dev, *rest), scored = batch
                if rest:
                    prop_cached = rest[0]
            else:
                x, y, scored, *rest = batch
                if rest:
                    prop_cached = rest[0].to(device, non_blocking=True)
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                scored_dev = scored.to(device, non_blocking=True)
"""
    proposed_batch = """            prop_cached = spread_encoded = None
            if args.pin_memory:
                (x, y, scored_dev, *rest), scored = batch
            else:
                x, y, scored, *rest = batch
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                scored_dev = scored.to(device, non_blocking=True)
            if args.spread_aux_mode != "original":
                spread_encoded = rest.pop()
                if not args.pin_memory:
                    spread_encoded = spread_encoded.to(device, non_blocking=True)
            if rest:
                prop_cached = rest[0] if args.pin_memory else rest[0].to(device, non_blocking=True)
"""
    proposed = replace_once(proposed, original_batch, proposed_batch)
    proposed = replace_once(
        proposed,
        "                sp_lab, br_lab, cl_lab = structure_labels(y, sequence_class(y))\n",
        "                sp_lab, br_lab, cl_lab = structure_labels(y, sequence_class(y))\n"
        '                if args.spread_aux_mode == "current_known":\n'
        "                    sp_lab = spread_encoded.clamp_max(N_SP_AUX - 1).long()\n",
    )
    original_struct = """                if args.aux_struct > 0:
                    o = out.float()
                    wv = sw[:, sl].reshape(-1)
                    ce = 0.0
                    for a0, a1, lab in ((struct_at, struct_at + N_SP_AUX, sp_lab), (struct_at + N_SP_AUX, struct_at + N_SP_AUX + 3, br_lab),
                                        (struct_at + N_SP_AUX + 3, struct_at + N_SP_AUX + 7, cl_lab)):
                        ce = ce + F.cross_entropy(o[..., a0:a1].reshape(-1, a1 - a0), lab[:, sl].reshape(-1), reduction="none")
                    loss = loss + args.aux_struct * (ce * wv).sum() / wv.sum().clamp_min(1e-8)
"""
    proposed_struct = """                if args.aux_struct > 0:
                    loss = loss + args.aux_struct * structural_loss(
                        out, sp_lab[:, sl], br_lab[:, sl], cl_lab[:, sl], sw[:, sl], struct_at, N_SP_AUX,
                        None if spread_encoded is None else spread_encoded[:, sl])
"""
    proposed = replace_once(proposed, original_struct, proposed_struct)
    ast.parse(proposed)
    return original, proposed


def main() -> None:
    """Write unapplied source patch plus exact before/after provenance."""
    original, proposed = proposed_source()
    support = (HERE / "trainer_support.py").read_text()
    patch = "".join(
        difflib.unified_diff(
            [],
            support.splitlines(keepends=True),
            fromfile="/dev/null",
            tofile="b/src/wnn/spread_aux.py",
        )
    )
    patch += "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            proposed.splitlines(keepends=True),
            fromfile="a/scripts/train_rnn.py",
            tofile="b/scripts/train_rnn.py",
        )
    )
    (HERE / "integration.patch").write_text(patch)

    manifest = {
        "applied": False,
        "trainer_base_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "trainer_proposed_sha256": hashlib.sha256(proposed.encode()).hexdigest(),
        "support_source_sha256": sha256(HERE / "trainer_support.py"),
        "patch_sha256": sha256(HERE / "integration.patch"),
        "features/model/outputs_changed": False,
        "spread_loss_denominator": "original sw.sum in both arms",
        "inherited_label_options": False,
        "FT_defaults": "original proxy supervision in BOTH arms",
    }
    (HERE / "integration_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
