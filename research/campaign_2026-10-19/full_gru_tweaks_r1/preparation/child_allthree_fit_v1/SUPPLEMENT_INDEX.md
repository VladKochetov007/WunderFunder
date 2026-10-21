# Final fixed arm: conditional local preparation

The [original packet README](README.md), [recipe binding](ALLTHREE_RECIPE_BINDING_V1.json), [binder](bind_allthree_fit_v1.py), [template](ALLTHREE_LAUNCH_TEMPLATE_V1.py), [focused fixtures](test_bind_allthree_fit_v1.py), and [local check receipt](LOCAL_CHECKS_V1.json) describe a conditional source packet. The [independent source review](ALLTHREE_SOURCE_REVIEW_V1.json) assesses those exact bytes without invoking a fit or renderer.

Only the NLL runname and base→ema_sparse representation argv values change; parent@ep3 initialization, NLL objective and WMSE/chunk compatibility flags, dictionary/cache/normalizer/order/target and six-epoch budget remain fixed. Actual NLL acceptance and exact stop are absent, so the final executable body remains absent. Both receipt validators require the actual NLL acceptance's original launch-acceptance SHA44a708fc. No profile replay, retuning, queue, B staging, GPU admission or quality comparison is implied.

Four focused tests passed through direct-file invocation. The earlier module-name test command failed before collection because the folder name contains a hyphen; [local checks](LOCAL_CHECKS_V1.json) preserve that invocation note. It consumed no remote support or scientific fit. Fresh serial gates and exact-body review remain required after actual NLL acceptance and stop.
