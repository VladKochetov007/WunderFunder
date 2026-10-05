# Parameter-count correction

The historical `activation_optimizer/ACTIVATION_NEXT_DESIGN.md` contains an
incorrect H64 total of 61,826. The live H64 two-layer GRU `RNNModel` has
**61,058** trainable parameters: input projection 2,752 + two-layer GRU 49,920
+ concatenation head 8,386. E474 computes the live count and asserts 61,058 for
every arm. The separately stated H256 count of 932,354 was correct.
