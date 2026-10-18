from __future__ import annotations

import torch
import torch.nn.functional as F

from wnn.distillation import four_family_weighted_mse
from wnn.nn import weighted_mse_loss
from wnn.outcomes import N_CODES, N_SP_AUX, readout, sequence_class
from wnn.spread_aux import structural_loss
from wnn.training.initializer import final_initializer_distillation_target
from wnn.training.state import TrainingRun
from wnn.training_controls import blended_target


def main_loss(run: TrainingRun) -> None:
    """Compute the original main objective, including distillation and its running normalizer."""
    args = run.options.args
    ce_w = run.batch.ce_w
    codes = run.batch.codes
    device = run.options.device
    family_at = run.network.family_at
    family_targets = run.batch.family_targets
    initializer_teacher = run.network.initializer_teacher
    inputs = run.batch.inputs
    loss = run.batch.loss
    main_valid = run.batch.main_valid
    moments = run.network.moments
    norm_ema = run.progress.norm_ema
    nxt_all = run.batch.nxt_all
    nxt_label = run.batch.nxt_label
    objective = run.optimization.objective
    out = run.batch.out
    out_ema = run.batch.out_ema
    s = run.batch.s
    sl = run.batch.sl
    td_a = run.batch.td_a
    teach = run.batch.teach
    y = run.batch.y
    if moments is not None:
        ce = F.cross_entropy(
            out.float().reshape(-1, N_CODES), codes[:, sl].reshape(-1), reduction="none"
        )
        wv = ce_w[:, sl].reshape(-1)
        loss = (ce * wv).sum() / wv.sum().clamp_min(1e-08)
        if args.ce_mse > 0:
            pred = readout(out, *moments.tables(), tilted=args.ce_tilt)
            loss = loss + args.ce_mse * weighted_mse_loss(pred, y[:, sl], main_valid[:, sl])
    elif teach is not None:
        yc = y[:, sl].float().clamp(-2, 2)
        if args.distill_alpha_far is None:
            a = args.distill_alpha
        else:
            rows = torch.arange(sl.start, sl.start + yc.shape[1], device=yc.device)[None]
            u = (nxt_label[:, sl] - rows).float().clamp_min(0)
            a = (
                args.distill_alpha
                + (args.distill_alpha_far - args.distill_alpha)
                * (1 - torch.exp(-u / args.distill_alpha_tau))
            )[..., None]
        label = yc
        if args.distill_boot_h:
            rows = torch.arange(sl.start, sl.start + yc.shape[1], device=yc.device)[None]
            ahead = rows + args.distill_boot_h
            alive = (nxt_label[:, sl] > ahead) & (ahead < teach.shape[1])
            idx = (
                ahead.clamp(max=teach.shape[1] - 1)
                .expand(yc.shape[0], -1)[..., None]
                .expand(-1, -1, 2)
            )
            label = torch.where(alive[..., None], teach.gather(1, idx).float().clamp(-2, 2), yc)
        if initializer_teacher is not None:
            tgt = final_initializer_distillation_target(y[:, sl], teach[:, sl])
        else:
            tgt = (1 - a) * label + a * teach[:, sl].float().clamp(-2, 2)
            if args.distill_reanchor:
                tgt = blended_target(
                    y,
                    teach,
                    nxt_label,
                    sl,
                    args.distill_alpha,
                    args.distill_boot_h,
                    args.distill_reanchor,
                )
        w = yc.abs() * main_valid[:, sl][..., None]
        loss = (w * (out[..., :2].float() - tgt) ** 2).sum() / w.sum().clamp_min(1e-08)
        if family_targets is not None:
            family_prediction = (
                out[..., family_at : family_at + 8].float().reshape(*out.shape[:-1], 4, 2)
            )
            family_target = (
                family_targets[:, sl].float().clamp(-2, 2).reshape(*out.shape[:-1], 4, 2)
            )
            family_loss = four_family_weighted_mse(family_prediction, family_target, w)
            loss = loss + args.distill_family_weight * family_loss
    elif args.td_alpha > 0:
        yc = y[:, sl].float().clamp(-2, 2)
        s_end = min(s + args.chunk, inputs.shape[1])
        jg = torch.clamp(nxt_all[:, sl], max=s_end) - 1
        rows = torch.arange(s, s_end, device=device)[None]
        teach_t = torch.gather(
            out_ema[..., :2].float(), 1, (jg - s).clamp(min=0)[..., None].expand(-1, -1, 2)
        )
        tgt = torch.where((jg > rows)[..., None], (1 - td_a) * yc + td_a * teach_t.clamp(-2, 2), yc)
        w = yc.abs() * main_valid[:, sl][..., None]
        loss = (w * (out[..., :2].float() - tgt) ** 2).sum() / w.sum().clamp_min(1e-08)
    elif args.loss_norm == "running" and args.loss == "wmse":
        yc = y[:, sl].float().clamp(-2, 2)
        w = yc.abs() * main_valid[:, sl][..., None]
        wsum = w.sum().detach()
        norm_ema = wsum if norm_ema is None else 0.99 * norm_ema + 0.01 * wsum
        loss = (w * (out[..., :2].float() - yc) ** 2).sum() / norm_ema.clamp_min(1e-08)
    else:
        loss = objective(out[..., :2], y[:, sl], main_valid[:, sl])
    run.batch.loss = loss
    run.progress.norm_ema = norm_ema


def auxiliary_loss(run: TrainingRun) -> None:
    """Add the declared auxiliary objectives in the original reduction and summation order."""
    args = run.options.args
    if (
        getattr(args, "research_feature_adapter_only", False)
        and getattr(args, "research_feature_adapter_objective", "inherited") == "target-only"
    ):
        return
    aux = run.batch.aux
    br_lab = run.batch.br_lab
    cl_lab = run.batch.cl_lab
    kth = run.batch.kth
    loss = run.batch.loss
    main_valid = run.batch.main_valid
    mask_at = run.network.mask_at
    mask_rows = run.batch.mask_rows
    n_out = run.network.n_out
    next_at = run.network.next_at
    nw = run.batch.nw
    nx_rows = run.batch.nx_rows
    nx_up = run.batch.nx_up
    objective = run.optimization.objective
    out = run.batch.out
    out_ema = run.batch.out_ema
    quote_at = run.network.quote_at
    quote_label = run.batch.quote_label
    scored_dev = run.batch.scored_dev
    sign_at = run.network.sign_at
    sl = run.batch.sl
    sp_lab = run.batch.sp_lab
    spread_encoded = run.batch.spread_encoded
    struct_at = run.network.struct_at
    sw = run.batch.sw
    tier_at = run.network.tier_at
    tier_label = run.batch.tier_label
    tier_weight = run.batch.tier_weight
    y = run.batch.y
    if args.arch == "gated" and args.gate_ce > 0:
        difference = y[:, sl, 0].float() - y[:, sl, 1].float()
        gate_weight = main_valid[:, sl].float() * difference.abs().clamp_max(2.0)
        gate_bce = F.binary_cross_entropy_with_logits(
            out[..., n_out].float(), (difference > 0).float(), reduction="none"
        )
        loss = loss + args.gate_ce * (gate_bce * gate_weight).sum() / gate_weight.sum().clamp_min(
            1e-08
        )
    if args.aux_struct > 0:
        loss = loss + args.aux_struct * structural_loss(
            out,
            sp_lab[:, sl],
            br_lab[:, sl],
            cl_lab[:, sl],
            sw[:, sl],
            struct_at,
            N_SP_AUX,
            None if spread_encoded is None else spread_encoded[:, sl],
        )
    if args.aux_next > 0:
        o = out.float()
        wv = nw[:, sl]
        mse = (o[..., next_at] - (nx_rows[:, sl] - 3.0) / 2.0) ** 2
        bce = F.binary_cross_entropy_with_logits(
            o[..., next_at + 1], nx_up[:, sl], reduction="none"
        )
        loss = loss + args.aux_next * ((mse + bce) * wv).sum() / wv.sum().clamp_min(1e-08)
        for j, (kr, ku, kok) in enumerate(kth):
            kw = main_valid[:, sl].float() * kok[:, sl].float()
            col = next_at + 2 + 2 * j
            kmse = (o[..., col] - (kr[:, sl] - 4.0) / 2.0) ** 2
            kbce = F.binary_cross_entropy_with_logits(o[..., col + 1], ku[:, sl], reduction="none")
            loss = loss + args.aux_next * ((kmse + kbce) * kw).sum() / kw.sum().clamp_min(1e-08)
    if args.aux_tier > 0:
        ce = F.cross_entropy(
            out[..., tier_at : tier_at + 4].float().reshape(-1, 4),
            tier_label[:, sl].reshape(-1),
            reduction="none",
        )
        tw = tier_weight[:, sl].reshape(-1)
        loss = loss + args.aux_tier * (ce * tw).sum() / tw.sum().clamp_min(1e-08)
    if args.aux_mask > 0:
        ce = F.binary_cross_entropy_with_logits(
            out[..., mask_at].float(),
            scored_dev[:, sl].float(),
            pos_weight=out.new_tensor(8.0),
            reduction="none",
        )
        mw = mask_rows[:, sl]
        loss = loss + args.aux_mask * (ce * mw).sum() / mw.sum().clamp_min(1.0)
    if args.aux_sign > 0:
        clipped = y[:, sl].float().clamp(-2, 2)
        difference = clipped[..., 0] - clipped[..., 1]
        sign_weight = main_valid[:, sl].float() * difference.abs().clamp_max(2.0)
        sign_bce = F.binary_cross_entropy_with_logits(
            out[..., sign_at].float(), (difference > 0).float(), reduction="none"
        )
        loss = loss + args.aux_sign * (sign_bce * sign_weight).sum() / sign_weight.sum().clamp_min(
            1e-08
        )
    if args.aux_quote > 0:
        label = quote_label[:, sl].float()
        quote_weight = (
            main_valid[:, sl].float()[..., None]
            * (sequence_class(y) == 3)[:, None, None]
            * torch.isfinite(label)
            * y[:, sl].float().clamp(-2, 2).abs().mean(-1, keepdim=True)
        )
        quote_error = (out[..., quote_at : quote_at + 2].float() - torch.nan_to_num(label)) ** 2
        loss = loss + args.aux_quote * (
            quote_error * quote_weight
        ).sum() / quote_weight.sum().clamp_min(1e-08)
    for j, (ya, va) in enumerate(aux):
        loss = loss + args.aux_weight * objective(
            out[..., 2 + 2 * j : 4 + 2 * j], ya[:, sl], va[:, sl]
        )
    if args.ema_teacher > 0:
        cw = main_valid[:, sl].float()[..., None]
        consistency_scale = 1.0
        if args.ema_metric_weight:
            cw = cw * y[:, sl].float().clamp(-2, 2).abs()
            consistency_scale = 2.0
        loss = loss + args.ema_teacher * consistency_scale * (
            cw * (out[..., :2].float() - out_ema[..., :2].float()) ** 2
        ).sum() / cw.sum().clamp_min(1e-08)
    run.batch.loss = loss
