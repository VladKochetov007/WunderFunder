# Native TPU dispatch status

The final native FP32-anchor/BF16-profile packet is built and independently checked, but **no new TPU session was accepted**. The Kaggle CLI returned exit0 with `Kernel push error: Maximum batch TPU session count of 1 reached.` Raw push receipt and root rejection review are retained. There is no measured TPU/GPU speed ratio.

The exact older private `vladkochetov07/e463-tpu-real-data-reload-v7-20261008` probe remains QUEUED; metadata confirms TPUv5e8 and kernel ID137741265. Its status, metadata and output-list responses do not provide a cancellable session ID. The kernel ID and per-kernel version number are not substituted. The output list is empty; that alone does not establish whether a script ever ran. No cancellation or blind retry occurred.

The installed SDK accepts cancellation only by session ID. The public interface limitation is also documented in [Kaggle CLI issue1169](https://github.com/Kaggle/kaggle-cli/issues/1169); a separate issue makes a conflicting version-number assertion, which we do not rely on. No connected local browser debugging session was found. Root asked the user to stop the queued probe in Kaggle or provide its session ID, while local and GPU work continue.

The3600second provisional hold remains reserved for this exact ready packet, alongside1650seconds for the old queued probe; neither is measured consumption. A slot change requires a new identified retry receipt before another push. Source/recipe/metric bytes remain frozen. [Final root admission](ROOT_TPU_V4_KERNEL_ADMISSION_R1.json), [raw provider receipt](TPU_V4_PROVIDER_PUSH_RECEIPT_R1.json), [rejection review](TPU_V4_SLOT_REJECTION_REVIEW_R1.json).
