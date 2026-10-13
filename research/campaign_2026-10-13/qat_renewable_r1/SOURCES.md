---
kind: method
status: proposed
created_utc: 2026-10-13T15:50:49Z
updated_utc: 2026-10-13T15:50:49Z
hypothesis_ids: [H-QAT-W4-20261013, H-QAT-A8-20261013, H-QAT-LSQ-20261013]
experiment_ids: []
evidence_paths: [research/campaign_2026-10-13/qat_renewable_r1/PLAN.md]
---

# QAT source components and limits

## Question and method

Which published quantization components motivate controlled experiments in the existing native recurrent predictor? These three primary arXiv records were opened on 2026-10-13; this note records source claims and proposed transfers, not local experimental gains.

| Primary source | Relevant component | Proposed local test and limitation |
|---|---|---|
| He et al., 2016, [Effective Quantization Methods for Recurrent Neural Networks](https://arxiv.org/abs/1611.10176), v1 | Quantizing gates/interlinks of LSTM/GRU and balanced weight quantization | H-QAT-A8: isolate recurrent matvec activation rounding with FP32 state. Their PTB/IMDB results do not establish our financial-task gains or require adoption of their entire quantizer. |
| Fasoli et al., 2022, [Accelerating Inference and Language Model Fusion of Recurrent Neural Network Transducers via End-to-End 4-bit Quantization](https://arxiv.org/abs/2206.07882), v1 | Full-model weight/activation QAT and network-specific quantization schemes | H-QAT-W4: uniform versus mixed recurrent precision, followed by activation-aware training. The paper estimates hardware acceleration by simulation; no CPU speed gain follows for our unpack-to-byte transport. |
| Esser et al., 2020, [Learned Step Size Quantization](https://arxiv.org/abs/1902.08153), v3 | Learn quantizer step sizes using scaled gradient estimates | H-QAT-LSQ: compare learned row scales against fixed row scales with identical fit budget and exact scale export. Published image-classification results are motivation, not recurrent-task evidence. |

## Evidence and interpretation

[E419](../../campaign_2026-10-04/E419_RECOVERED_QAT_REPORT.md) is the actual local precedent: matched weight-only Q8 QAT did not establish a gain over fine-tuning/PTQ. It evaluated represented weights with floating activations; native dynamic activation rounding remains a separate source of error. Existing native `q8vnni*` also preserves the I16 input projection while QAT uses floating projection. The first wave keeps that serving choice fixed across controls and candidates.

## Decision and next step

Follow the [plan](PLAN.md): exact code/scale export and deployment baselines, matched W4/mixed fits, then separately controlled activation and learned-scale extensions. Read full method equations before implementing those extensions. Exact storage size, recurrent error and actual CPU runtime must be measured independently.
