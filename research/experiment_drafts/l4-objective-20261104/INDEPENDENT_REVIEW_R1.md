# Completed teacher comparison: independent review

The independent recipe/evidence critic `l4_teacher_recipe_critic_20261104`
reviewed the exact complete report, membership/update counts, paired arithmetic,
and both train/validation and gradient figures. Verdict: PASS for reporting the
negative result; this is not candidate promotion.

Report SHA256: `4adb4aefea7d89d83ab04d86a8ee786caf94e2e3975f2d518181c08184366549`.

The fixed zero-based epoch3 control mean is 0.6851345441432545. Clip5 has
paired mean delta -0.0000586696910538; late auxiliary annealing has
-0.0002126917911784. Neither meets the preregistered +0.001 teacher gate.
The actual selector receipt now accompanies the report and says `no_students`.
No optional student was fitted or new target cache produced for these arms.

All four fresh fits completed 25,760 updates each, 103,040 total, exit0.
All six fresh/control teachers peaked descriptively at epoch2; the primary
endpoint remains frozen epoch3. Two matched seeds on repeatedly inspected
development validation do not establish significance, a general model ceiling,
or expected public improvement.

The report's trainer-internal `dev` to `vfold` alias is qualified by exact
official validation membership, not by the display label. This repair changed
report interpretation, not fitting members, checkpoints or evaluation rows.
The diagnostic failure remains a failure and provides no gradient-conflict
evidence. The separate PR61 Transformer continuation is outside this result.
