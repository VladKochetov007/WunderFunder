You are an independent research ideator for Wunder Connectome Challenge2. Our supplied sequences have 112 anonymous numerical feature channels and 2 regression targets. Inference is causal row-by-row, deterministic, with sequence-state reset; future features/labels are unavailable. Evaluation uses a fixed scored/prediction mask and global weighted Pearson with labels/predictions clipped +/-2, label-magnitude weights. Training clipping is independent of official evaluation. Own independently trained weights and independently implemented code are mandatory. Only supplied permitted data and permitted public pretrained sources. Existing causal event/delta/recency views and GRU teachers/students are a basis, not a ceiling. Training teachers may exceed final CPU submission envelope and be compressed later. Local 6 CPU/20GBRAM/5GBVRAM; Kaggle GPUs/TPUs and bounded owned Vast workers available. Avoid OOM. Current active families include heterogeneous recurrent teacher distillation, causal TCN, event-feature wider GRU, online feature dynamics, block memory, and causal heuristic window features. Do not restrict ideas to these families. This is directed-source transfer, not copying geology code or claiming geology leaderboard gains transfer. No fits or tools needed; output your final public proposal text only, not private reasoning. You cannot view another ideator's proposals.

Read the whole supplied writeup carefully. Generate at least 6 DISTINCT plausible mechanism cards, including one unconventional small-component transfer; make a serious effort to extract at least one usable detail even when its headline seems irrelevant. Do not fabricate applicability. Consider geometry/reframing, distributions/loss, aggregation/multiscale, calibrated feature channels, particle belief states with rare jumps, invariant augmentation, conditional diversity/quality gates, negative experiments and hardware. Future-looking smoothing or symmetric U-Nets cannot be used causally as-is: propose a causal replacement or explicitly training-only privileged teacher with audited fitting ancestry. Never infer exact semantics/geological invariants for anonymous features. Current/previous weak standalone models can become useful components; specify complete endpoint controls.

For EACH card provide: unique local ID; exact writeup section/component (short quote <=12 words or paraphrase); structural analogy; intervention and implementable components; assumptions; causal leakage/validity risks; predicted observable; cheapest bounded discriminating test and matched control; falsifier (scope exact recipe, not entire family); approximate cost tier with no invented timing; likely parent/interaction links; level-2 successors. Give a diverse top-3 ordering justified by information and expected long-term usefulness, not fabricated score priors or automatic zero-crossing-CI rejection. No numerical expected-public promises. Do not copy source code. Deliver standalone markdown cards, plus a compact integration table.

SOURCE follows (user-supplied full text; reference https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/writeups/1st-place-solution):

Thanks to Kaggle and the competition hosts for organizing such an interesting competition. I was very lucky to survive the leaderboard shake-up and win my first 1st place gold medal.

I'd also like to thank everyone who shared ideas publicly, especially @tuckerarrants , who motivated me to explore neural networks early in the competition, and whoever originally shared the particle filter baseline (unfortunately I couldn't trace the original author).

Most of the code was written with Codex using GPT-5.5/5.6. The repository contains many implementation details, so this write-up only covers the ideas that I believe contributed most of the performance gains. For the full implementation, I recommend giving the code to an AI coding agent and asking it to explain the details. The repository also contains many inactive experimental features, which can safely be ignored.
Task Formulation and Loss

I formulate the problem as a 2D alignment task and use cross-entropy loss as the main objective.

The alignment grid is:

    Horizontal well: 345 positions (downsampled by 32), consisting of 1,024 visible plus 10,000 target region.
    Typewell: 400 positions covering ±100 ft around the last visible TVT with 0.5-ft resolution.

Throughout this write-up, all TVT values are relative to the last visible TVT, which serves as the anchor.

The training target is an exponentially smoothed probability distribution centered around the ground-truth alignment and normalized along the typewell dimension.

Additional losses:

    Huber loss on the expected TVT path, computed from the predicted probability map and grid TVT values.
    GR penalty loss: mean(probability × grid_GR_gap), which provides a small additional improvement.

Model

I use a standard 2D U-Net with a ConvNeXt backbone:

    timm/convnext_small.in12k_ft_in1k_384
    Standard residual blocks in the decoder.

Replacing LayerNorm in ConvNeXt with BatchNorm consistently works better, although BF16 training is required to avoid NaN losses.

For downsampling and upsampling, simple average pooling + interpolation performs better than learnable alternatives.
Features
Typewell

    GR
    GR is NaN
    TVT

Typewell GR is calibrated using the visible region by aggregating (TVT, GR) into TVT bins, interpolating back to the typewell axis, and blending with the original GR values.
Horizontal well

For each downsampled bin:

    GR mean
    GR nan rate
    GR standard deviation
    GR slope
    Last − first GR
    Quadratic fitting coefficients
    Quadratic fitting residual RMSE
    Visible-region TVT mean

Typewell × Horizontal interactions

    |typewell_GR - horizontal_GR|
    |typewell_TVT - visible_TVT|

Coordinate based

    z_diff

Particle-filter features

    2D particle probability heatmap
    |typewell_TVT - PF_TVT|

XY-neighbor features

    Predicted TVT difference
    Predicted TVT
    |typewell_TVT - predicted_TVT|

When particle-filter channels are present (which already contain accumulated TVT information), using accumulated TVT performs better than using only TVT differences.
Particle Filter

This component was largely developed with the help of AI coding agents since I was not very familiar with particle filters. The standalone local CV is approximately 7.4 RMSE.

Compared with the public baseline, I believe the major improvements come from:

    Allowing low-probability large jumps
    Calibrated typewell GR
    FFBSi smoothing
    Blending diverse cfg profiles
    Updating particles in bins (64 samples) instead of raw resolution

There are also many smaller implementation details that can be found in the released code.
XY Neighbor Information

Assume the local geological surface satisfies

    S = TVT + Z + C

and S is locally linear, then

    ΔTVT = aΔX + bΔY − ΔZ

where (a, b) are estimated by weighted least squares using (x,y) neighbors.

The implementation includes anisotropic distance, singular-case handling, regularization, and several other engineering details. However, these contribute only marginal improvements over the simple formulation above, so I won't discuss them further.

The standalone local CV is approximately 11.4 RMSE.

Adding XY-neighbor information consistently improved local CV but hurt the public leaderboard, so I performed several analyses.

    XY-neighbor prediction alone scores 12.9 LB which lies in high prob region for 50 wells sample.
    I replaced XY based predictions with GR + z_diff predictions whenever neighborhood statistics exceeded the 95th percentile of the training distribution.

The neighborhood statistics include:

    Mean neighbor distance
    10th percentile neighbor distance
    Prefix/visible neighbor weight ratio
    Distance(query, neighbor center) / average distance(neighbor, neighbor center)
    Average abs cosine corr((x_diff,y_diff),(x_nbr_diff,y_nbr_diff))

None of these statistics explained the leaderboard degradation, and I believe they sufficiently describe neighborhood quality.

Therefore, I attribute the discrepancy to inconsistent labels and chose to trust the local CV, which consistently improves by about 0.3 RMSE.

These same neighborhood quality statistics are also used later in the ensemble.
Data Augmentation

The two most important augmentations are Z-shift and GR transform.
Z-shift

Randomly sample a TVT path while keeping TVT + Z unchanged.

The sampled path is generated using block bootstrap on real TVT differences.

GR is regenerated by matching the typewell using:

    TVT + TVT noise

where the TVT noise consists of smoothed white noise.

Rare fault jumps are also simulated, introducing offsets in TVT + Z.
GR transform

Apply

    GR' = a × GR + b

to the typewell GR, forcing the model to rely more on shape than absolute values.
Other augmentations

    Reverse path (reverse traversal while keeping part of the beginning visible)
    MD stretching
    2D channel masking
    GR noise shift (shift residual (GR - matched_typewell_GR) within each well)
    Tail cropping
    Sequential masking along the typewell axis
    Typewell GR jitter (consistent transformation of GR channels)
    PF rotate/shift (corrupt PF channels to reduce shortcut learning)

Ensemble

For wells without reliable XY-neighbor information (approximately 10%), I use models without XY-neighbor channels and include the z_diff channel instead.

Below, "default" refers to models using GR, TVT, and z_diff channels without particle-filter or XY-neighbor features. Models also differ slightly in augmentation and training configurations. (for version difference details see version_comparsion.md in project_files section)
Model 	Features 	Weight (general / no XY) 	CV (3 seeds × 5 folds avg) 	Public LB 	Private LB
Weighted ensemble 	— 	— 	4.627 	5.980 	5.639
0719_V1 	Default 	0.07 / 0.40 	5.09 	5.648 	6.130
0729_V3 	+ Particle filter 	0.00 / 0.20 	5.53 	6.202 	6.768
0801_V1 	Default 	0.07 / 0.40 	5.16 	5.723 	5.884
0724_V1 	+ XY-neighbor channels 	0.28 / 0.00 	4.86 	6.095 	5.831
0801_V2 	+ XY-neighbor channels 	0.28 / 0.00 	4.80 	6.166 	5.937
0803_V2 	+ XY-neighbor channels + Particle-filter channels 	0.28 / 0.00 	5.00 	6.185 	5.778
What didn't work

    Transformer-based backbones

    In my experiments, CNN-based backbones consistently performed better, and only ConvNeXt contributed to the final solution.

    Pretraining on simulation data first

    Training with simulated/augmented data and real data jointly worked better for me than a two-stage approach that first trains on simulation data.

    Using PF features in the final single model

    Initially, PF features provided a significant improvement, reducing CV RMSE from 7.7 to 6.7. However, after adding more augmentations and switching to stronger pretrained backbones, PF features no longer improved my best single model and I kept them in my final ensemble for diversity.
    Considering that my PF-only approach achieved around 7.4 RMSE in CV, a stronger PF-based method may still provide additional gains.

    Fancy loss weighting strategies

    I experimented with various loss weighting schemes, but none of them provided consistent improvements.

    Removing visually bad-labeled wells

    Removing wells with visually poor labels did not result in stable gains.

    Scaling up to larger models or higher resolution

    Further increasing model size or input resolution did not improve performance in my experiments.

Code

The inference notebook is available here:

https://www.kaggle.com/code/w5833946/submit-reproduce

for reproduce training check: https://github.com/IAmAValidUsername/kaggle_ROGII_1st_place_solution_Ruby
Author
Ruby

w5833946
Share
Project Links
submit_reproduce

2 months ago · 58 upvotes

Kaggle Notebook

bronze medal

Public
Project Files

    train_png_typewell_map.csv
    B_submission_model_Ruby.zip
    version_comparsion.md

Download All
Citation

Ruby. 1st Place Solution. https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/writeups/1st-place-solution. 2026. Kaggle
26
34 Comments
7 appreciation comments
 

This comment will be made public once posted.
Post Comment
Igor Kuvaev
Competition Host

Posted 2 months ago

@w5833946 Ruby, thank you for the participation and write up!
I was filling you and you were leading most of the time. Great result! Congrats!
Reply
React
Ruby
Topic Author

Posted 2 months ago

· 1st in this Competition

Thanks a lot! It was a pleasure competing in this challenge, and huge thanks to you for hosting such a wonderful and interesting competition!
Reply
React
Cody_Null

Posted 2 months ago

· 2237th in this Competition

Hi @w5833946 Great work throughout the competition, I am having a tough time replicating your results. Could I trouble you to share the training code as well? I have a feeling what I am missing in replicating your solution is what held me back throughout this competition.
Reply
React
Ruby
Topic Author

Posted 2 months ago

· 1st in this Competition

The training code is also included in the datasets attached to the notebook, just start training from seq_NN_main.py without passing 'submit-mode' after set data path cfg properly. I just uploaded a B_submission_model zip which provides some guidelines for reproduce.
Reply
React
Cody_Null

Posted 2 months ago

· 2237th in this Competition

Thank you so much! Hadnt caught that one!
Reply
React
depressionkid

Posted 2 months ago

· 5581st in this Competition

Awesome, thank you for sharing. For me is a whole bunch of topics to investigate
Reply
React
Alerdo Ballabani

Posted 2 months ago

· 3241st in this Competition

Well done Ruby, you should be so proud of this achievement!
Reply
React
testofschool

Posted 2 months ago

· 654th in this Competition

this is something, congrats!
Reply
React
Hugo F Valdez

Posted 2 months ago

· 4941st in this Competition

Great approach! Congrats !!
Reply
React
TERMINUS

Posted 2 months ago

· 457th in this Competition

Good solution! Congratulations!
Reply
React
Rustam Bazarbayev

Posted 2 months ago

· 2245th in this Competition

Conratulations!
Reply
React
xsong2020

Posted 2 months ago

· 952nd in this Competition

Great work and congratulations, GM!
Reply
React
LL

Posted 2 months ago

Congratulations master!
Reply
React
ForcewithMe

Posted 2 months ago

· 1793rd in this Competition

Solid and Clear. Congrats for gm @w5833946
Reply
React
khwaja Moula'Ali

Posted 2 months ago

· 5290th in this Competition

Congratulations Ruby
Reply
React
Michael Timbs

Posted 2 months ago

· 130th in this Competition

Thanks for this. Some lessons here for me. I closed some of these directions too early or by measuring the wrong thing. Very nice solution.
Reply
React
Ignasi Alemany

Posted 2 months ago

congrats! will now learn from your solution - my first serious competition! hopefully I can grind more in the next one
Reply
React
斬る

Posted 2 months ago

· 205th in this Competition

Congratulations on becoming a Grandmaster, and congrats on taking first place solo! This method is really cool.
Reply
React
Forrest_xlz

Posted 2 months ago

· 1528th in this Competition

beautiful method get beautiful score
Congratulations！
Reply
React
Connor Tynan

Posted 2 months ago

· 365th in this Competition

Congratulations Ruby! And thank you for sharing <3
Reply
React
Sasha Turutin

Posted 2 months ago

· 138th in this Competition

Thank you for sharing the code. Amazing work!
Reply
React
swordsman

Posted 2 months ago

· 525th in this Competition

恭喜，太厉害了，我以为一定是lightgbm相关模型呢。。。佩服。。
Reply
React
Tucker Arrants

Posted 2 months ago

· 26th in this Competition

Congratulations Ruby - GM with only solo golds is a remarkable achievement.
Reply
React
Ruby
Topic Author

Posted 2 months ago

· 1st in this Competition

Thanks for all your helpful discussions — they helped me a lot in this competition. I think you were just missing a bit of luck this time. Hope things go your way in the next one, and looking forward to seeing your great results again!
Reply
1
Thiago Munhoz da Nóbrega

Posted 2 months ago

· 430th in this Competition

congrats and thank you for the write up! to me, your approach was very creative.
Reply
React
Mohit

Posted a month ago

Niceee congratulations
Reply
React
skw1990

Posted 2 months ago

· 3844th in this Competition

@w5833946 can you share how the rationale of how you start the problem? like the thought process?
Reply
React

This comment has been deleted.
Appreciation (7)
k256.dev

Posted 2 months ago

· 6th in this Competition

congratulation 🎊
Muhammad Junaid

Posted 2 months ago

· 1775th in this Competition

congratulation 🎊
Behiye Bedir

Posted 2 months ago

· 1019th in this Competition

congratulations!
Homii_N

Posted 2 months ago

· 1323rd in this Competition

Congratulations…
Samer Attrah

Posted 2 months ago

· 5169th in this Competition

Congratulations
Arthur Kim

Posted 2 months ago

· 367th in this Competition

Congratulations!
GG Ayo (AyoGG)

Posted 2 months ago

· 75th in this Competition

congratulation 🎊
