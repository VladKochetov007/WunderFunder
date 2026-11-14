# Queued kernel and cancellation identity

The official SDK continues to report `QUEUED` for `vladkochetov07/wnn-q14-strict-q14s-20261114-r2-runner`; no execution log or worker output has appeared. Raw GetKernelSessionStatus JSON is `{}`, which the generated protobuf client maps to its default queued enum. GetKernel confirms private kernel ID144995259, current version1 and the expected owner/reference, but exposes no session ID. This does not identify the reason for the delay or prove any fit has started.

Read-only inspection found an SDK cancel-session method requiring an exact kernel_session_id. The CLI lacks a corresponding command. An authenticated API-session GET of the private webpage returned404 and did not disclose a run ID; private scratch holds the response and no credentials were logged. No cancel, delete, new kernel version or retry was issued.

The upstream [session-ID issue1169](https://github.com/Kaggle/kaggle-cli/issues/1169) describes the same missing reference-to-session mapping. [Issue1172](https://github.com/Kaggle/kaggle-cli/issues/1172) claims the push version number can be used as a session ID; this conflicts with the stronger identity limitation and does not bind our small per-kernel version1 to an exact global run. Root will not send cancellation to an unverified ID or delete the kernel to resolve the queue.

The proposed local fallback remains conditional on a verified terminal/cancelled cloud state; no duplicate panel is admitted by this note. Independent local fitting, fixed trajectory-mean preparation and native replacement evaluation continue. The cloud launch retains its bounded13,500-second provider timeout if it starts.
