# Fixed-six-epoch producer endpoints

All six fresh fits completed 38,760 updates and six full official validations.
The table uses the frozen sixth epoch, not a validation-selected checkpoint.
Each target specialist supplies only its trained target; its unused head is
excluded. No validation-fitted mixing weights are used.

| Fixed output | Seed203201 WP | Seed203202 WP | Mean WP | Mean paired delta |
|---|---:|---:|---:|---:|
| Shared control | 0.680506272 | 0.682647483 | 0.681576878 | 0 |
| Specialist t0 + control t1 | 0.680097461 | 0.681953093 | 0.681025277 | -0.000551601 |
| Control t0 + specialist t1 | 0.681000144 | 0.682193260 | 0.681596702 | +0.000019824 |
| Both specialists | 0.680591333 | 0.681498870 | 0.681045101 | -0.000531776 |

The t0-only replacement loses on both seeds. The tiny positive t1 replacement
mean combines opposite paired signs (+0.000493872 and -0.000454223), and gives
no clear evidence of a reliable improvement. These are arithmetic means of
same-seed complete outputs, not scores of a seed ensemble. The two-specialist
package costs two inference models and does not improve the mean here.

`fixed-e6-producer-grades-r1.json` preserves full precision and source/split
identities. The ROOT fit controller verified successful owned exits, completion,
coverage and model hashes. This is producer monitoring evidence on the repeatedly
exposed official valid set (1,873 sequences; 3,528,273 scored rows), not a public
submission or fresh confirmation.

Canonical final-checkpoint export/scoring is a separate verification step.
Its R1 receiver failed before export because resolving a virtualenv Python
symlink selected the system interpreter without NumPy. No training result was
lost. R2 preserves the virtualenv interpreter path and uses a separate output
namespace; its outcome is reported separately. No training is repeated.
