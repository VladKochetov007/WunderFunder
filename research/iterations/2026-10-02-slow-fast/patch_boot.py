from pathlib import Path
p=Path("scripts/train_rnn.py");s=p.read_text()
old='''                elif teach is not None:
                    yc = y[:, sl].float().clamp(-2, 2)
                    tgt = (1 - args.distill_alpha) * yc + args.distill_alpha * teach[:, sl].float().clamp(-2, 2)
'''
assert s.count(old)==1
new='''                elif teach is not None:
                    yc = y[:, sl].float().clamp(-2, 2)
                    label = yc
                    if args.distill_boot_h:
                        # stopped bootstrap (ported from VladKochetov007/WunderFunder main): while the label run outlives
                        # t+h, replace y_t by the teacher at t+h (training-time target only)
                        nxt_label = next_label_change(y)
                        rows = torch.arange(sl.start, sl.start + yc.shape[1], device=yc.device)[None]
                        ahead = rows + args.distill_boot_h
                        alive = (nxt_label[:, sl] > ahead) & (ahead < teach.shape[1])
                        idx = ahead.clamp(max=teach.shape[1] - 1).expand(yc.shape[0], -1)[..., None].expand(-1, -1, 2)
                        label = torch.where(alive[..., None], teach.gather(1, idx).float().clamp(-2, 2), yc)
                    tgt = (1 - args.distill_alpha) * label + args.distill_alpha * teach[:, sl].float().clamp(-2, 2)
'''
s=s.replace(old,new)
s=s.replace('    ap.add_argument("--seq-repeat",','    ap.add_argument("--distill-boot-h", type=int, default=0, help="stopped bootstrap distillation target (rows ahead)")\n    ap.add_argument("--seq-repeat",',1)
p.write_text(s)
