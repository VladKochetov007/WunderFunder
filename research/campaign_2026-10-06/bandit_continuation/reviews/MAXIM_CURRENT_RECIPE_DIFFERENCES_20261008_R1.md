# Current teammate-recipe differences after normalized public result

2026-10-08. User reported normalized own all-data seed3 ZIP scored **0.6653** (submission1n7tmww1,2007s/624MB). Public incumbent remains0.6660. This does not establish teammate0.6674 parity.

| Part | Teammate successful endpoint | Our scored normalized endpoint |
|---|---|---|
| Student architecture/core settings | GRU256x2,alpha0.7,2 additional epochs,LR1e-4,B16/C2000,aux0.3/0.3,family0.1 | Broadly matched |
| Teacher fitting coverage | Wave18 final manifest explicitly includes validation labels in component training | Familyteachers trained on cleanfold0; predictions cover allrows but that does not make teacherweights all-data trained |
| Warm parent | Seed-specific fbc1k parents seed5/6 | Shared own all-data seed3 parent; different earlier ancestry |
| Strong source | One fbc1k seed5 source at0.30 | Original normalized cache uses two own cleanfold strong sources at0.15 each |
| Baseline contribution | Organizer baseline0.0625 | Two own samefold raw baseline substitutes0.03125 each |
| Main teacher target | Unnormalized weighted sum with mass0.8125 | Normalized by0.8125, increasing teacher amplitude by1.230769 before downstream operations |
| Served endpoint | Equal seed5/seed6 student pair | Single seed3 student |
| Family target clipping | Raw family predictions | Current implementation raw too; historical mismatch was fixed |

Evidence: inspected frozen final_student_lane.sh and prepare_full_teachers.py; origin/research/creative-iteration-20260930 wave18 full-teacher-manifest.json; own cache metadata bd26_h230_fold0_teacher_family_all_rows_v1 explicitly declares clean_fold_fit_only; own normalized contract and public ZIP receipt.

These are supervision/ancestry/endpoint differences, not merely a hyperparameter sweep. Independently trained weights will always differ, but recipe-level matching remains necessary before a faithful replication claim. The public scores compare complete submissions, not an isolated normalization intervention. No single difference is proven to cause the gap.

Priority remains independently trained all-data teacher families, seed-specific parents, faithful unnormalized main target and a paired endpoint. The new history-swap ZIP is unscored and replaces the old strong-pair contribution in the main target; it still does not replace the cleanfold familyteachers. No public forecast.
