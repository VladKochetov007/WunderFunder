# S3 fitting-valid prediction export V1 — completed

Originalinvocation6c4365b925564e27af985ea833addcf6 ran22:56:51–23:00:24UTC, exit0, all1,405fitting-valid sequences in88progressrecords. Rootcaptured23:00:55/accepted beforeexactstop23:02:01. Togetherwith the [completedS3trainexport](../teacher_s3_train_export_v1/README.md), bothS3fittingpredictionparts are ready. This is inference, zero newscientificfits/qualitygrades; the completeT3mean cache and GRUcomparison remainpending.

## Evidence

- [Launchacceptance](LAUNCH_ACCEPTANCE.json), [rawlaunchresponse](STDOUT.raw.json), [transport](TRANSPORT.json), [admission](ADMISSION.json), [launch](LAUNCH.json). Extracts are semanticrawcopies, notB-formatbyteidentityclaims.
- [Terminalhealth](health_v1/STDOUT.raw.json) and [transport](health_v1/TRANSPORT.json), [rawterminalcapture](capture/STDOUT.raw.json), [capturetransport](capture/TRANSPORT.json), [sidecar](capture/PREDICTION.json), [nativeprogresslog](capture/unit.log).
- [Rootterminalacceptance](TERMINAL_ACCEPTANCE.json), [integerterminalfields](TERMINAL_EVIDENCE.json), [resourcereconciliation](RESOURCE_RECONCILIATION.json), [exactstopraw](exact_stop/STDOUT.raw.json) and [stoptransport](exact_stop/TRANSPORT.json).
- [Source-reviewedthinadapter](../../preparation/TEACHER_CACHE_S3_VALID_EXPORT_REVIEW.json) keeps immutablecode/profile, accepted/exact-stoppedS3train predecessor, unchangedfourmetadataJSONs and newvalidfileabsence.

## Scope/limits

AcceptedS3epoch2/own211-wideFP32normalizer/source88becca167files/18semanticpins. Exactordered1,405F0fit-valid groups match acceptedprovenance and are disjointfrom468evaluationgroups in the same valid-dataset namespace. Feature-only/rawmain2/strictFP32, no evaluationrows/labels/masks. NPY224,800,128Bstat-only/producerSHA0ba98e6df9d59fb3c301388ab06ff1d34875946b2809d2be6efb251c1fa0baef atB/research_tmp/fg19_t3_fit_predictions_s3_s4_s5_v1/pred_fg19_t3_h384x2_e3_s3_v1_valid.npy; sidecarSHA55ddc99fd7aa06ba3a9b309421b3f4642c826abf0a918ca9616b82020b19be9b. No independentarrayfinite/header/hash replay; streamedCPUcacheconsumer mustvalidatefinite/shape/alignment. No realpayloadcopiedtoMac.

Nativewall213.001799s/CPU211.056941s/reportedserviceMemoryPeak2,428,928B are distinctmeasurements. The smallservicefieldisnotfullruntimeRAM; noactiveCG/RSSsample wascaptured forthisexport, scope/valuegapunresolved/noinventedcause. EmptyterminalCG=>ownfilesNA/nohostfallback; parentnewmemoryevents0/populated{}. SameoneCPU/high7.5e9/max8e9/swap0/4GiBallocator+1GBreserve/runtime10800s/GPU asS3train. Nohealthyjobstop: exactstoponlyafterdurableterminalvalidation. Neverpoll/replaystoppedexport. NextgatedstageS4train, noautomaticqueue.
