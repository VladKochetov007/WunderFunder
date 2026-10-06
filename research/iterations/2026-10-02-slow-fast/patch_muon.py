from pathlib import Path
p=Path("scripts/train_rnn.py");s=p.read_text()
old='''    opt = torch.optim.AdamW([prm for prm in model.parameters() if prm.requires_grad], lr=args.lr, weight_decay=args.wd,
                            fused=bool(args.fused_adam))
    total = len(train_loader) * math.ceil(20_000 / args.chunk) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total,
                                                pct_start=0.05, final_div_factor=100)
'''
assert s.count(old)==1
new='''    total = len(train_loader) * math.ceil(20_000 / args.chunk) * args.epochs
    trainable = [(n, prm) for n, prm in model.named_parameters() if prm.requires_grad]
    if args.optimizer == "muon":
        # Muon on hidden 2-D matrices (GRU weights, hidden head layer); AdamW on input projection, output layer, biases
        is_muon = lambda n, prm: prm.ndim == 2 and not n.startswith(("inp", "head.2", "mean", "std"))
        mu = [prm for n, prm in trainable if is_muon(n, prm)]
        ad = [prm for n, prm in trainable if not is_muon(n, prm)]
        print(f"muon params {sum(x.numel() for x in mu)} adamw params {sum(x.numel() for x in ad)}", flush=True)
        opts = [torch.optim.Muon(mu, lr=args.lr * args.muon_lr_mult, weight_decay=args.wd, momentum=args.muon_momentum,
                                 adjust_lr_fn="match_rms_adamw"),
                torch.optim.AdamW(ad, lr=args.lr, weight_decay=args.wd, betas=(0.9, args.beta2), fused=bool(args.fused_adam))]
    else:
        opts = [torch.optim.AdamW([prm for _, prm in trainable], lr=args.lr, weight_decay=args.wd, betas=(0.9, args.beta2),
                                  fused=bool(args.fused_adam))]
    scheds = [torch.optim.lr_scheduler.OneCycleLR(o, max_lr=o.param_groups[0]["lr"], total_steps=total,
                                                  pct_start=args.warmup_frac, final_div_factor=100) for o in opts]
    opt, sched = opts[-1], scheds[-1]
'''
s=s.replace(old,new)
old2='''                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
'''
assert s.count(old2)==1
s=s.replace(old2,'''                for o in opts:
                    o.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
''')
old3='''                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
'''
assert s.count(old3)==1
s=s.replace(old3,'''                for o in opts:
                    scaler.unscale_(o)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                for o in opts:
                    scaler.step(o)
                scaler.update()
                for sc in scheds:
                    sc.step()
''')
s=s.replace('    ap.add_argument("--extra-exclude-fold",','''    ap.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw")
    ap.add_argument("--muon-lr-mult", type=float, default=1.0)
    ap.add_argument("--muon-momentum", type=float, default=0.95)
    ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--extra-exclude-fold",''',1)
p.write_text(s)
