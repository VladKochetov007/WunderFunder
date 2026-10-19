# Archived Clean T3 Muon H384×2 evidence

The archived R6 terminal package contains a successful historical individual
fit, `clean_t3_f0_muonall_h384_e3_s3_r5`, with seed 3, H384×2 GRU, three
epochs, fixed epoch-2 checkpoint, batch 32, chunk 1000, Muon, one-cycle
schedule, WMSE, FP16, and a learned propensity dependency. Its archived
command and output are useful evidence about the historical teacher recipe;
they do not make it a runnable current-source recipe or a clean reusable parent.

The current canonical optimizer CLI accepts `--muon-scope rnn|hidden`; it does
not accept the archived `--muon-scope all`. A strict-FP32 run that changes this
scope without defining its semantics would not be a faithful port. Historical
fold and cache identities also differ from current Beleriand bindings, and the
archived propensity model/cache dependencies are not established by the R01
portable-source receipt. No current teacher command is therefore admitted by
this note.

The seed-3 result is one identified completed fit, not the strongest
individual point score: the inspected seed-5 result is higher (0.67969849 vs
0.67898918 on the recorded WP scale). Those historical scores do not select a
new seed or establish cross-fold performance. The campaign's selected strong
family is the Main T3 student recipe; the teacher trio and a source-qualified
warm parent remain separate clean prerequisites.

The bounded profile currently prepared in this folder exercises only the
published strict-FP32 propensity implementation on actual fold-0 fit-valid
rows. It is not a teacher fit, does not port Muon-all semantics, and does not
establish a full-GRU training runtime or score. Any teacher or warm-parent
recipe requires a separately source-bound semantic port and review before
admission.
