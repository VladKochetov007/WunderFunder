# Rust source patches

The GELU patch vectorizes the existing approximation. The cache-order patch changes independent member scheduling/prefetch only. Their exact blob base, tests, actual library identity and bitwise/runtime result are described in the [engineering index](../README.md) and [receipts](../../evidence/README.md). Neither patch meets the serving budget yet.
