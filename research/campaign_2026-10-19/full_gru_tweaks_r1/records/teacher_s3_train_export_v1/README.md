# S3 fitting-train prediction export V1 — completed

The original invocation `48732136d70548b2903f3426bfedf656` ran22:12:29–22:39:16UTC, exit0, all10,535fitting-train sequences written in659batches. Rootcapturedcompactmetadata22:50:39, accepted before exactstop22:53:03. This is one completed frozen-teacher inference export, zero new scientific fits and no quality grade. All three teacher fits remain accepted/stopped; no old training or prediction was replayed.

## Actual evidence

- [Native admission](ADMISSION.json), [native launch](LAUNCH.json), [raw launch response](STDOUT.raw.json) and [transport](TRANSPORT.json). Compact extracts are semantically equal to rawresponse, not claimed byte-identical to Bformat.
- [Successful terminal captureV2](capture_v2/README.md), [root terminal acceptance](TERMINAL_ACCEPTANCE.json), [integer-normalized terminal evidence](TERMINAL_EVIDENCE.json), [resource reconciliation](RESOURCE_RECONCILIATION.json), [exact stop](exact_stop/README.md).
- [Producer sidecar](capture_v2/PREDICTION.json), [orderedfit membership excerpt](capture_v2/FIT_MEMBERSHIP.json), [immutableteacher provenance](capture_v2/TEACHER_PROVENANCE.json), [native progress log](capture_v2/unit.log).
- [Source-review beforelaunch](../../preparation/TEACHER_CACHE_S3_TRAIN_EXPORT_REVIEW.json) and [independentactual launch review](../../preparation/TEACHER_CACHE_S3_TRAIN_EXPORT_ACTUAL_REVIEW.json).
- [Initial startup](startup_health_v1/README.md), [human-requested halfwaystatus](health_20261019T2226Z/README.md), [terminaldiscovery](health_20261019T2248Z/README.md). These are dated observations; no finishedunitpolling.

## Scope and numerical boundary

Source88becca/167files, exact18semanticruntimepins, acceptedS3epoch2/checkpoint and its own211-wideFP32mean/std. Rawmain2FP32 prediction construction, exactF0fittrain10535order/feature-only/no evaluation rows/labels/masks. NPYpath B/research_tmp/fg19_t3_fit_predictions_s3_s4_s5_v1/pred_fg19_t3_h384x2_e3_s3_v1_train.npy,1,685,600,128Bstat-only, producerwholefileSHA218e3c41e91ff45e691b85e6999412841ca05c68aebdfc25c1c7659b0cc6bec5. The capture did not independently reopen/headercheck/rehash or verify finiteness of the wholearray; the later streamedCPUcacheconsumer must perform finite/shape/alignment checks. Source-qualifieddtype/shape are not an independent payload audit. Realpayload staysB; noMacarray/checkpoint.

Envelope1CPU/high7.5e9/max8e9/swap0/4GiBallocator+1GBreserve/runtime10,800s, oneownedA4000. Actualnativewall1607.235567s/CPU1606.223317s/systemdMemoryPeak3,054,235,648B; sampledCG/CUDA/RSS fields remain separate observations. EmptyterminalCG=>ownfilesNA/nohostrootfallback; parentnewevents0/populated{}.

[V1 metadata capture guard](capture/README.md) failed becausewholeF0split1.388MB exceeded1MBoutputcap. Astat-onlydiagnostic identifiedit; V2preserves1MB/outputand4MBaggregate, sending only aSHA-boundmembershipexcerpt. No model/inference replay or capsearch. Unknownlaunchoutcome never retries/restarts. Stoppednativeunit must not be polled or replayed. S3fit-valid/S4/S5exports and CPUmean×.8125oncecache remain later serial gates, noautomaticqueue.
