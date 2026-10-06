for p in $(pgrep -u $USER -f "2026-10-02-slow-fast/bip.py [01]"); do kill $p 2>/dev/null; done
cd ~/WunderFunder-creative && rm -f cache/propcache_bip_f*_*
cat >> research/iterations/2026-10-02-slow-fast/README.md <<"MD"
- MASK SMOOTHER (bip.py v2: frozen causal prop + backward-GRU correction, zero init): fold 0 AUC .9218 vs causal .9217, Brier .0515 vs .0521 after 4 epochs.
  Future rows add ~nothing to scoring-mask prediction. REFUTED; stopped.
MD
echo stopped
