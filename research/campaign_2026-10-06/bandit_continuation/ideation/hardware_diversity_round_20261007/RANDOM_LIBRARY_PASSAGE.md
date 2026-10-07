# Random user-library passage

- **Selection frame:** the distinct PDF paths listed in
  `research/ideas/library_ideation_footprint_2026-10-06.json` that existed at
  selection time; frozen as `RANDOM_SELECTION_POOL.txt` (14 files).
- **Procedure:** sort the unique absolute paths, then select one with
  Python's `random.Random(20261007).choice(paths)`; no topic-conditioned
  filtering. Pool digest is SHA-256 of UTF-8 `"\n".join(paths)` with no final
  newline: `899a7c60ac7f7d8ed14a09384e036b22a90456ca11ad045e9c7ff58c157fce62`.
- **Draw:** `/home/vlad/Documents/Trading/Naive/TWAP & VWAP - More important
  than you think.pdf`, Vertox, “TWAP & VWAP — More important than you think,”
  28 August 2023, PDF page 4/7, subsection “Technical Details.”
- **Exact excerpt (21 words):** “We can actually make computation easy since
  we can calculate VWAP recursively and don’t need to recalculate everything
  at every timestep.”
- **Transfer attempt:** maintain normalized numerator/denominator state or
  weighted sufficient statistics incrementally, rather than materializing
  every historical row. This is an algorithmic analogy only. The article is a
  popular execution post, not a primary research paper; it supplies no evidence
  about Challenge2 labels, forecasting quality, volume semantics, or the
  official metric.
- **Why it was selected:** it was selected by the recorded uniform draw before
  adapting it to the problem. The selection was not edited to choose this
  passage after reading candidate ideas.
