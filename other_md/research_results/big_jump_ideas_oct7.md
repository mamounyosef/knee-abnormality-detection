# Big-jump ideas, 7 Oct (forum re-read + web)

Goal: 0.943 -> ~0.96. Only ideas NOT already run (see runs_summary.md / runs_timeline.md).

## What the forum says now (read 7 Oct, kaggle discussion)

- Best single-model thread (735304): 5-fold ResNet-50 @224 0.954 (17th), single 224px 0.952, 5-fold 0.955 (42nd),
  CoAtNet 224 single fold 0.950, Qwen3.5-2B 0.950, tennogh (10th) 0.943 @288 "OOF pseudo-labels well correlated with LB",
  Tucker Arrants (12th) 0.943 5-fold @224. Small models at low resolution beat our 512 px singles (best 0.934),
  so the gap is labels, not architecture. Chris Deotte: with pseudo-labels every single model matches the ensemble.
- Raymond Yuen (42nd): several weak label sources -> teachers -> pseudo-labels -> final model; "different teachers
  help different targets"; label API cost < $5.
- Tricks thread (745861): Claude text labels -> single 0.942; image-aware LLM labels only 80-90% right on 800 studies;
  examples in the prompt -0.004 CV / +0.002 LB.
- External data (743416, 741819, 734109): YYama (9th) and Tucker (12th) say relevant external knee MRI could give a
  "dramatic improvement". Host: allowed only if free and with no escalated/institutional review or agreement;
  KneeCoT banned for its agreement with the dataset institution. Tucker: virtually every useful set fails
  (institutional sign-off or no-derivatives).
- Manual labels of TRAIN images (740718): participants read it as allowed (no host answer).

## Candidates (new to us)

1. ROUND-2 PSEUDO-LABELS from a DIVERSE out-of-fold teacher. v4 = v3 50% + ConvNeXt folds 25% + MII folds 25% was ONE
   round from two architectures on v3 labels. Round 2: 5-fold CV of the 3-4 most different members on v4 labels
   (ConvNeXt 2.5D, MII, DINOv3 ConvNeXt, SwinV2), their OOF mean = the teacher, new labels, then 5-fold students.
   Evidence: the forum consensus above; Noisy Student (arxiv 1911.04252) iterates teacher -> student.
2. TRANSDUCTIVE self-training on the hidden test set inside the 9 h notebook: ensemble pseudo-labels the ~1,300 test
   studies, one fast member fine-tunes on train + test pseudo-labels, then predicts. Allowed (no human labels).
   Simulate offline first: pseudo-label val, fine-tune, score val. Evidence: weak-medium (TransBoost, CFT
   arxiv 2401.16991).
3. fastMRI knee DICOM (fastmri.med.nyu.edu): 10,000 clinical knee MRIs with OUR five sequences, unlabeled. Noisy
   Student on 14k studies is the largest possible jump, BUT access is an agreement with NYU ("internal research or
   education only", no transfer of derived data files): same reason KneeCoT was banned. High disqualification risk.
   fastMRI+ boxes (CC BY 4.0, 22 pathologies) sit on the k-space subset, same images, same risk.
4. Hosted frontier VLM (Claude/GPT) image-grounded verdicts on the most disputed cells only (not done; MedGemma was
   local). Forum: 80-90% accurate; use as one more teacher vote, never hard labels.
