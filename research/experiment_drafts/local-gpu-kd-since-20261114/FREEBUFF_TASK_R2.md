# Task 2: how to get more out of distillation here

Same setting as task.md. New measured fact: a cold 1M-param GRU student trained on 0.5*label + 0.5*teacher
prediction (teacher = one wider GRU, valid score 0.6914) reaches 0.6905 vs 0.6867 for labels only, in 4 epochs,
and unlike the label-only run it does not overfit in the last epoch. Loss is |y|-weighted MSE with event-age
row weights. Teacher predictions exist only on the training sequences. Three teachers are available
(0.6914, 0.6878, 0.6874; mutual correlation ~0.99).

Already closed here: correlation losses, categorical heads, heteroscedastic NLL, density-ratio reweighting,
per-event auxiliary heads (already used), stateful TBPTT (already used).

Give 6 concrete ideas to push the student to or above its teacher, each with mechanism, exact change, falsifier:
think about what the student target should be (per-row mixing weight, teacher disagreement, residual targets),
what to do on rows where teacher and label disagree, self-distillation rounds, using the teacher on unlabeled
or perturbed inputs, and curriculum over epochs. No generic advice. Write to ideas2.md.
