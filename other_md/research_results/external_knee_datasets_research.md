# External knee MRI datasets for the RSNA Knee Abnormality Detection competition

Research date: 2026-10-04. Method: web search plus direct fetches of primary pages (dataset cards, agreements, GitHub/Hugging Face/Zenodo/Mendeley records, papers). Anything I could not confirm from a primary source is marked "unverified". Quotes are copied from the fetched primary text unless marked "summary of fetched page" (some sites are JavaScript-rendered or blocked, so only a summary was available).

Fetch problems worth knowing: the NIMH Data Archive (NDA) OAI pages (nda.nih.gov/oai) and the NIA Aging Research Biobank are JavaScript-only and returned no text; the original OAI Online "Data Use Agreement" (UCSF, HIPAA Limited Data Set form) could only be read partially (page 1 of 4 on a mirror). The NDA "Data Use Certification" (DUC) that now governs OAI downloads was read in full from the NDA PDF.

---

## 1. Executive summary

### Bottom line

There is no external dataset that is both SAFE (open license, no agreement, equally accessible) and high value for our weak findings (lateral OA, patellofemoral OA, synovitis, lateral meniscus tear). The data that would help most (OAI, with image-graded MOAKS scores) is free but sits behind an institutional data use certification, which is exactly the "equally accessible to all participants" problem the rules warn about. The SAFE datasets (KMAR-50K, ACL-PCL) carry no useful labels for us.

### Ranked shortlist (by practical value to us, then by safety)

| Rank | Dataset | Value for our weak findings | Prize-safety | One-line verdict |
|---|---|---|---|---|
| 1 | OAI (Osteoarthritis Initiative) incl. MOAKS readings (also covers the FNIH 600-knee subset) | High: image-graded cartilage by subregion, osteophytes, effusion-synovitis, Hoffa-synovitis, bone marrow lesions, meniscus, popliteal cyst | RISKY (lean avoid) | Best science, worst access. Needs NDA account plus sponsoring institution with a Federalwide Assurance, a signed DUC, publication duties, and a destroy-derivatives clause. |
| 2 | fastMRI + fastMRI+ annotations | Low to medium: meniscus (663 subjects), effusion (142), cartilage defects, edema, cysts; coronal only; single reader | RISKY | "Internal research or educational purposes only". Annotations are MIT, images are not. |
| 3 | KMAR-50K (Mendeley, Sichuan hospitals) | Low: no pathology labels; usable only for self-supervised or restoration pretraining | SAFE (license shown as CC BY 4.0 on Mendeley; one source says CC BY-NC-ND, see section 5) | Only clean, open option. Expected gain on our findings is small and unproven. |
| 4 | KneeMRI (Rijeka, ACL) | Very low: ACL only, 917 sagittal PD fat-sat volumes | RISKY (CC BY-NC-ND 4.0) | We are already strong on ACL. |
| 5 | MRNet (Stanford) | Very low: ACL, meniscus, abnormal; report-derived labels like ours | RISKY (non-commercial, no derivatives, no sharing) | Not worth the agreement. |
| 6 | MeniMV (3,000 exams, horn-level meniscus grades) | Potentially useful for lateral meniscus tear, but release and license unverified | unverified | Check whether it is actually downloadable. |
| 7 | SKM-TEA | Very low: 155 qDESS scans | RISKY (Stanford non-commercial agreement) | Too small and 3D qDESS. |

NOT USABLE: Radiopaedia, MedPix (case collections), IEEE DataPort knee set, K2S (data withdrawn after the challenge), IWOAI 2019 (by request only), KNOAP2020 (test labels are incident OA, not available as MRI training data), MOST (closed to new requests since 2021), KneeCoT (hospital-signed ethical use agreement, CC-BY-NC-4.0, commercial use prohibited).

### Recommendation

1. Default: use no external data. The expected gain is uncertain, the rules around "equally accessible" and non-commercial terms create real disqualification risk, and our training/test label mismatch (report-derived vs image-graded panel) is not something these sets fix cheaply.
2. If we want a low-risk experiment: pretrain or continue self-supervised training of the backbone on KMAR-50K (open license), then fine-tune on our data. Treat it as a small bet. It adds MRI texture and contrast diversity but no labels.
3. If we want the high-upside experiment (auxiliary heads for OA per compartment, synovitis, effusion, bone marrow lesion from OAI MOAKS): first ask the Kaggle hosts in the competition discussion whether OAI via NDA counts as "publicly available and equally accessible", because NDA access needs institutional sponsorship. Do not start without a yes. Also confirm we personally can get NDA access (unverified for independent individuals).
4. Never use or upload anyone's redistributed copy of OAI images (for example the Hugging Face mirror `charwisreno/OAI-Knee-MRI`, whose license field says "Other (users must follow the Osteoarthritis Initiative Data Use Agreement)"). Redistribution looks inconsistent with the NDA DUC.

---

## 2. Comparison table

Prize-safety legend: SAFE = open license, public download, no restrictive agreement. RISKY = free but behind a DUA, registration with terms, research-only or non-commercial terms, no-redistribution, or institutional approval. NOT USABLE = paid, closed, withdrawn, or credentials not equally accessible.

| Dataset | Modality / sequences | Size | Labels (mapped to our 12) | Label source | Access | License / terms | Safety | URL |
|---|---|---|---|---|---|---|---|---|
| OAI | 3T knee MRI, 4 US sites: sagittal 3D DESS, coronal 2D IW TSE, sagittal 2D IW TSE fat-sat, coronal 3D FLASH; also X-ray | 4,796 subjects, over 431,000 visits, about 26.6 million images (per re3data summary); MRI is multiple TB | MOAKS subset: effusion-synovitis (to effusion), Hoffa-synovitis (to synovitis), BML (to bone contusion, partial), cartilage + osteophytes per subregion (to 3 OA compartments), meniscus (to meniscus tears), popliteal cyst (to Baker's), ACL/PCL. No fracture, MCL unverified | Image-graded by radiologist readers (MOAKS). KL grades/JSN from X-ray central reads | NDA account, RAS login, DUC cosigned by institutional official with FWA | NDA DUC (quoted below) | RISKY | https://nda.nih.gov/oai |
| FNIH OA Biomarkers subset | Same OAI 3T MRI | 600 knees (600 participants), baseline and 24 months | MOAKS (same items as above) | Image-graded readers | Through OAI/NDA | Same as OAI | RISKY | https://fnih.org/our-programs/osteoarthritis-initiative/ (page returned 403; facts from published papers) |
| MOST | Knee MRI in a 3,026-person cohort | 3,026 at baseline | MOAKS reported in papers | Image-graded | MOST Online stopped accepting requests 26 Feb 2021 (per search summary of MOST site) | MOST Online DUA (research and/or educational) | NOT USABLE | https://mostonline.ucsf.edu/ |
| 3DReasonKnee labels (OAI-derived) | DESS 3D, sag/cor/ax | 7,970 volumes, 494k to 512k Q/A rows | MOAKS grades, bboxes, masks (nnU-Net) | MOAKS from OAI readings plus 450 clinician hours | Labels on Hugging Face, images require OAI | CC BY 4.0 (labels only) | RISKY (images needed) | https://huggingface.co/datasets/rajpurkarlab/3DReasonKnee |
| OAI-ZIB / OAIZIB-CM | 3D DESS, right knee baseline | 507 knees | Bone/cartilage segmentation only | Manual segmentation | Zenodo/HF | OAIZIB-CM: CC BY-NC 4.0; original ZIB license unverified | RISKY | https://zenodo.org/records/14934086 |
| MRNet | 1.5T/3T: coronal T1, coronal T2 FS, sagittal PD, sagittal T2 FS, axial PD FS | 1,370 exams | Abnormal, ACL tear, meniscus tear (any) | Extracted from clinical reports | Register, emailed link | Stanford Research Use Agreement (quoted) | RISKY | https://stanfordmlgroup.github.io/competitions/mrnet/ |
| KneeMRI (Rijeka) | 1.5T Siemens Avanto, sagittal PD fat-sat | 917 volumes (3.45 GB data) | ACL: healthy/partial/complete | Manually annotated | Open Zenodo download | CC BY-NC-ND 4.0 | RISKY | https://zenodo.org/records/14789903 |
| fastMRI knee | 3T/1.5T; raw k-space coronal PD and PD-FS only; DICOM has all five sequences | 1,398 raw (697 PD, 701 PD-FS) per site; 10,000 clinical DICOM; 1,172 annotated | None in base set | n/a | Application with affiliation and agreement | "internal research or educational purposes only" | RISKY | https://fastmri.med.nyu.edu/ |
| fastMRI+ knee | Coronal single series per exam | 16,154 boxes, 13 study labels, 22 categories, 1,172 studies | meniscus tear, cartilage defect, effusion, cysts, edema, fracture/contusion, ACL/MCL sprain | One MSK radiologist (6 yrs), image-graded | GitHub CSV; images need fastMRI | Annotations MIT; images fastMRI agreement | RISKY | https://github.com/microsoft/fastmri-plus |
| SKM-TEA | 3D qDESS, 3T | 155 scans (900 GB compressed) | 16 pathologies as boxes (meniscus, cartilage lesions), 6-tissue masks | Boxes extracted from radiology reports, masks manual | Redivis account | Stanford non-commercial agreement | RISKY | https://aimi.stanford.edu/datasets/skm-tea-knee-mri |
| IWOAI 2019 | OAI 3D DESS | 176 images (88 patients) | Cartilage, meniscus masks | Single segmenter | Request from organizer | n/a | NOT USABLE | https://pubs.rsna.org/doi/10.1148/ryai.2021200078 |
| K2S | 3D CUBE fat-sat | 300 train patients | 6-compartment segmentation | Radiologist approved | Not shared after challenge | CC BY-NC-ND | NOT USABLE | https://k2s.grand-challenge.org/Data/ |
| KNOAP2020 | OAI MRI + X-ray | test 423 knees | Incident symptomatic radiographic OA | Outcome | Registration; training from OAI | n/a | NOT USABLE | https://knoap2020.grand-challenge.org/Data/ |
| KMAR-50K | 1.5T/3T, PD-TSE, PD-FSE, T1 and more (list truncated in my fetch) | 1,190 patients, 1,444 pairs, 62,506 images | No pathology labels (motion-artifact pairs) | n/a | Open Mendeley | CC BY 4.0 on Mendeley page | SAFE | https://data.mendeley.com/datasets/xw7mrg7ntg/6 |
| ACL-PCL (East West University) | Axial, coronal, sagittal T2, sagittal PD-FS | 970 augmented images, 640 by 640 | ACL/PCL normal vs torn (image level) | unverified | Open Mendeley | CC BY 4.0 | SAFE (tiny) | https://data.mendeley.com/datasets/fjns2xzhrr/1 |
| MeniMV | Sagittal and coronal T2WI-FS | 3,000 exams, 750 patients, 6,000 scans | Anterior and posterior horn grades 0 to 3 per meniscus | Double-validated by clinicians | Release/link unverified | unverified | unverified | https://arxiv.org/abs/2512.18437 |
| KneeCoT | 3D knee MRI | 25,000 volumes, 1.05 TB | Meniscus, ligaments, cartilage, bone marrow changes, effusion; reports and VQA | Expert reports | Hospital-signed ethical use agreement | CC-BY-NC-4.0 plus terms | NOT USABLE | https://huggingface.co/datasets/YiHui0124/KneeCoT |
| OAI mirror on HF (`charwisreno/OAI-Knee-MRI`) | OAI volumes as .npy | 92,313 volumes, 3.53 TB | CSV metadata | OAI | HF login with contact info | "Other" (follow OAI DUA) | RISKY (avoid) | https://huggingface.co/datasets/charwisreno/OAI-Knee-MRI |
| Chen 2018 knee X-ray KL set (Mendeley/Kaggle) | X-ray (not MRI) | 8,260 images (5,778 + 826 + 1,656 splits) | KL grade | OAI central readings | Open | CC BY 4.0 | SAFE license, wrong modality | https://data.mendeley.com/datasets/56rmx5bjcr/1 |
| IEEE DataPort "Knee" | Sagittal MRI | 15.78 MB | Segmentation | Surgeons | Subscription | not specified | NOT USABLE | https://ieee-dataport.org/documents/knee |
| Radiopaedia | Case collections | n/a | n/a | n/a | Website | Terms prohibit ML use | NOT USABLE | https://radiopaedia.org/terms |
| MedPix | Case collections | n/a | n/a | n/a | Website | CC BY-NC-SA 3.0 per search summary; ML policy unverified | NOT USABLE | https://catalog.data.gov/dataset/medpix |
| MURA | Upper-extremity X-ray, no knee | 40,561 images | Normal/abnormal | Radiologist | Agreement | Stanford RUA | Not relevant | https://stanfordmlgroup.github.io/competitions/mura |
| MedMNIST | No knee MRI in v2 | n/a | n/a | n/a | Open | CC BY 4.0 (DermaMNIST NC) | Not relevant | https://medmnist.com/ |

---

## 3. Detailed notes per dataset

### 3.1 OAI (Osteoarthritis Initiative)

What it is: a multi-center, longitudinal, prospective cohort of knee osteoarthritis, hosted since January 2019 by NIH at the NIMH Data Archive. The NDA repository holds 4,796 subjects, over 431,000 visits and about 26.6 million images (figures from the re3data record and search summaries of the NDA page). Source: https://nda.nih.gov/oai and https://www.re3data.org/repository/r3d100011843

Modality and sequences (3T Siemens Trio, 4 sites), verified from the OAI MRI design paper summary: sagittal 3D DESS with water excitation (TE/TR 4.7/16.3 ms, 160 slices, 0.7 mm slices), coronal 2D intermediate-weighted (IW) TSE (35 slices), sagittal 2D IW TSE with fat suppression (37 slices), coronal 3D FLASH with water excitation. There is no axial 2D TSE series. Match to our setup: the sagittal IW TSE fat-sat is the closest analogue to our sagittal fluid-sensitive series, the coronal IW TSE to our coronal plain series. Source: https://www.oarsijournal.com/article/S1063-4584(08)00223-9/fulltext

Labels available (verified at the file-listing level): the OAI releases semi-quantitative MOAKS readings (HF repo `rajpurkarlab/3DReasonKnee` ships the OAI files named `kMRI_SQ_MOAKS_BICL00/01/03/06` and the FNIH variants `kMRI_FNIH_SQ_MOAKS_BICL00/01/03`). Per the 3DReasonKnee card, MOAKS covers: bone-marrow lesions (BMLs) by subregion, cartilage lesions (size and depth), osteophytes, meniscal extrusion and morphology, Hoffa-synovitis, effusion-synovitis, ACL and PCL, patellar tendon, bursitis, popliteal cysts, ganglion cysts. The expansion of "BICL" is unverified (I assume a core lab). Number of readable knees is subset-dependent: FNIH is 600 knees at baseline and 24 months; one paper trained on 1,628 knees with MOAKS effusion-synovitis labels from the public OAI release. Whole-cohort KL grade and joint space narrowing come from X-ray central readings (unverified file names).

Mapping to our 12 findings (my judgement, not a published mapping):

| Our finding | OAI/MOAKS signal | Quality of match |
|---|---|---|
| Joint effusion | MOAKS effusion-synovitis grade | Good |
| Synovitis | MOAKS Hoffa-synovitis grade (and effusion-synovitis) | Moderate (Hoffa is one site) |
| Medial/lateral/PF OA | Cartilage lesion size/depth per subregion, osteophytes per site, JSN on X-ray, KL | Good, and the lateral and patellofemoral subregions are explicit |
| Bone contusion | BML | Partial: degenerative BMLs differ from traumatic bruises |
| Medial/lateral meniscus tear | MOAKS meniscus morphology per horn | Moderate |
| Baker's cyst | Popliteal cyst | Good |
| ACL tear | MOAKS ACL | Moderate |
| MCL tear, fracture | Not graded (unverified for MCL) | None |

Label source: radiologist image readings. That is the type of label we want (our test labels are image-based panel reads).

Access process: NDA account via Research Auth Service (eRA Commons, Login.gov or PIV/CAC credentials), then a Data Use Certification. Source: https://nda.nih.gov/oai (summary).

Exact DUC clauses (read from the NDA PDF https://nda.nih.gov/ndapublicweb/Documents/NDA+Data+Access+Request+DUC+FINAL.pdf, OMB 0925-0667):

- Institutional sponsorship: "Recipients with Institutional sponsorship acknowledge that access, if provided, is for research that is approved by the institution with which they are affiliated, which must be operating under an active Federal Wide Assurance (FWA) issued by the Department of Health & Human Services, Office for Human Research Protections (OHRP)." The NDA terms document adds: "Investigators and institutions seeking data from NDA will be expected to meet data security measures and will be asked to submit a Data Use Certification, which is cosigned by the investigator and the designated Institutional Official(s) at the NIH-recognized sponsoring institution with a current Federal Wide Assurance (FWA)."
- Purpose: "Recipients agree to use data for scientific investigation, scholarship or teaching, or other form of research and research development." It does not name competitions or commercial use. Term 14 mentions "any resulting commercial product(s)" only in the endorsement disclaimer. So a DUA clause forbidding competitions is not present, but access is tied to institution-approved research.
- No resale or onward sharing: "controlled-access data obtained through the request, in whole or in part, may not be sold to any individual at any point in time for any purpose." and "not to distribute controlled-access data to any entity or individual not identified in the request without appropriate written approvals from the NIH."
- Derived data: "Recipients agree to not publish or distribute any derived data that could aid in the re-identification of any of the study participants... Before subject-level derived data are distributed outside of an NDA Study, approval should be sought from the Help Desk."
- Destruction: "Upon close-out, the Recipient's institution and the Recipient agree to destroy all copies, versions, and data derivatives of the data retrieved from NIMH Data Archive, on both local servers and hardware, and if cloud computing was used, delete the data and cloud images from cloud computing provider storage". Unverified whether trained model weights count as "data derivatives"; this is the clause most likely to collide with a code competition that ships weights and later open-sources them.
- Publication duty: "Recipients agree to create and share an NIMH Data Archive Study ... for each publication, computational pipeline, or other public disclosure of results from the analysis of data accessed in the NIMH Data Archive". Open-sourcing a winning pipeline plausibly triggers this.
- Acknowledgement: "Recipients agree to acknowledge the appropriate NIMH Data Archive data repository and the relevant Digital Object Identifier(s) (DOI)... in any and all oral and written presentations, disclosures, and publications".
- Security: data must be managed per "NIH Security Best Practices for Users of Controlled-Access Data". That conflicts with uploading images to Modal or Kaggle without checking the practice document (unverified).
- Older OAI Online DUA (UCSF): described as a HIPAA "Limited Data Set" agreement with publication acknowledgement text ("The OAI is a public-private partnership comprised of five contracts..."). Full text not read.

Prize-safety: RISKY. Reason: free, but not "equally accessible" if sponsorship by an FWA institution is required, and the destroy/publish/NDA-Study duties add uncertainty about our own submission being open-sourced. NOT USABLE if we cannot obtain institutional sponsorship.

Practical value: highest of everything found, for OA compartments, synovitis, effusion.

Risks: 3D DESS (our series are 2D TSE; only sagittal IW TSE FS and coronal IW TSE match), 3T only, older cohort (45 to 79) with OA or OA risk, few acute injuries, no axial series, MOAKS label granularity differs from our binary labels, subregion scores must be collapsed into binary compartment OA by a rule we invent.

### 3.2 FNIH OA Biomarkers Consortium subset

600 knees from 600 OAI participants (297 progressors, 303 non-progressors), MOAKS read at baseline and 24 months on 3T MRI, 599 of 600 with both MOAKS and cartilage morphometry. Source: https://bmcmusculoskeletdisord.biomedcentral.com/articles/10.1186/s12891-016-1310-6 and https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5105263/. It is a nested subset of OAI, so access and license terms are the OAI ones. The FNIH page returned HTTP 403, so no separate FNIH terms were verified. Best-curated MOAKS set but small (600 knees), progression-enriched (medial JSW loss), so skewed to medial OA.

### 3.3 MOST (Multicenter Osteoarthritis Study)

3,026 adults aged 50 to 79 at baseline, knee MRI assessed with MOAKS. Per the MOST site summary: "As of Friday, February 26, 2021, MOST Online will no longer accept requests for MOST Public Use Limited Datasets and/or Images, due to the end of funding and closeout of MOST." Early-cycle data may be requested via the NIA Aging Research Biobank (https://agingresearchbiobank.nia.nih.gov/studies/most/, controlled access, page content not readable). MOST DUA (https://mostonline.ucsf.edu/Account/UserAgreement), summary of fetched page: use limited to "research and/or educational purposes"; recipient "agrees not to identify the information contained in the Limited Data Set or to contact the individual"; must "cite and acknowledge MOST and its funding sources". Classification: NOT USABLE (closed to new requests; images not mentioned in the agreement).

### 3.4 3DReasonKnee (OAI-derived labels, CC BY 4.0)

Rajpurkar lab Hugging Face dataset, license field `cc-by-4.0`. Not gated. From the raw README: "`/OAI/` - Raw DICOM files from OAI dataset", "`/labels/` - Contains taxonomymatch csv files with MOAKS scores", "`/images/` - Contains the preprocessed images", "`/masks.zip` ... segmentation masks ... given by nnU-Net". Under "How to Use": "Please download the original raw DICOM files from https://nda.nih.gov/oai/query-download". The HF API listing shows 5,375 files (about 6.8 GB): segmentation NIfTI for 100 knees, label CSVs, split files, Q/A JSONs. I found no image files in the repo listing, so the images still need OAI access. License statement: "Our 3DReasonKnee dataset is released under the Creative Commons Attribution 4.0 International (CC BY 4.0) license" (commercial use allowed with attribution). That license covers their annotations, not OAI images. Paper: 7,970 volumes, 494k quintuples, 450+ clinician hours. https://arxiv.org/abs/2510.20967

Use: gives a ready-made table of OAI MOAKS labels and a train/val/test split for DESS, handy only if we also have OAI images. Safety: labels SAFE, images RISKY.

### 3.5 OAI-ZIB and OAIZIB-CM

OAI-ZIB: 507 OAI exams with manual femur/tibia bone and cartilage segmentations, 3D DESS water excitation, 160 sagittal slices, right knee baseline; hosted via Zuse Institute Berlin (zib.de/research/pubdata, hosted on Zenodo; exact license unverified, my fetch of the ZIB page gave no license text). Derived OAIZIB-CM (Zenodo 14934086 and HF `YongchengYAO/OAIZIB-CM`) is CC BY-NC 4.0, 12.7 GB, "The CC BY-NC license restricts use to non-commercial purposes with proper attribution" (summary). Contains images, so it redistributes OAI images under NC terms, which sits uneasily with the NDA DUC (unverified legal question). No pathology labels, so useful only for cartilage/meniscus segmentation auxiliary heads. Classification: RISKY.

### 3.6 MRNet (Stanford)

1,370 exams: 1,104 abnormal, 319 ACL tears, 508 meniscal tears; labels "obtained through manual extraction from clinical reports" (report-derived, same noise type as ours). Sequences per the competition page: coronal T1, coronal T2 FS, sagittal PD, sagittal T2 FS, axial PD FS; 3.0 T (56.6 percent) or 1.5 T. Splits: 1,130 train, 120 tuning, 120 hidden test. Access: register, receive an emailed link, "you may not share the link". Dataset DOI https://doi.org/10.71718/rcbp-8c35 (Redivis via AIMI).

Agreement clauses, quoted from the Stanford ML Group agreement page source:
- "Permission is granted to view and use the MRNet Dataset without charge for personal, non-commercial research purposes only. Any commercial use, sale, or other monetization is prohibited."
- "YOU MAY NOT DISTRIBUTE, PUBLISH, OR REPRODUCE A COPY of any portion or all of the MRNet Dataset to others without specific prior written permission from the School of Medicine."
- "You must not modify, reverse engineer, decompile, or create derivative works from the MRNet Dataset."
- "...the School of Medicine may direct that the undersigned data recipient immediately return all copies of the MRNet Dataset and retain no copies thereof"

Judgement: personal non-commercial research only, no derivative works (arguable for trained weights), and a prize-bearing competition whose winners release under CC-BY-NC is a gray zone. Classification RISKY. Gain: only ACL and combined meniscus tear, which we already do well. Source: https://stanfordmlgroup.github.io/competitions/mrnet/ and the agreement at https://github.com/stanfordmlgroup/stanfordmlgroup.github.io/commit/a07223dfda18ebae2a09a8c43e11e7365c3ccc90.patch

### 3.7 KneeMRI (Štajduhar, Croatia)

917 12-bit grayscale volumes from a Siemens Avanto 1.5T at Clinical Hospital Centre Rijeka (2006 to 2014), sagittal proton density fat-suppressed, labeled healthy 690, partially injured 172, completely ruptured 55 (counts from the ELNet and Rijeka literature summaries). Zenodo record 14789903: "Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License (CC BY-NC-ND 4.0)", about 3.45 GB data (3.7 GB archive), open access. The Kaggle copy (`sohaibanwaar1203/kneemridataset`) is reported as CC BY-NC-ND 4.0 (search summary, not verified on the page). Original citation request: Štajduhar et al., Comput Methods Programs Biomed 140 (2017) 151-164. Classification RISKY (NC and ND). Value: ACL only, so negligible.

### 3.8 fastMRI knee and fastMRI+

fastMRI (NYU): per the site, 1,500+ fully sampled knee raw datasets (listed as 697 PD and 701 PD-fat-suppressed scans on the page, total 1,398) and 10,000 clinical knee DICOM studies at 3T and 1.5T. The raw set is coronal only; the DICOM set contains all five sequences including sagittal and axial (per the fastMRI RSNA AI paper). Access: application including institutional affiliation and proposed uses, then acceptance of the Data Sharing Agreement. Terms (summary of fetched page, phrase corroborated in search results): non-exclusive royalty-free license "to use the fastMRI Dataset for internal research or educational purposes only"; no sharing of download links; no selling or monetizing; no redistribution without prior permission; destroy all copies on completion; report violations within 5 business days; citations required. https://fastmri.med.nyu.edu/

fastMRI+ (Microsoft Research, Nature Scientific Data 2022): 1,172 knee studies (all knee raw datasets in train plus validation), each a single coronal series (PD or T2 style), 16,154 bounding boxes and 13 study-level labels for 22 categories, by one subspecialist MSK radiologist (6 years) on MD.ai. The paper warns the "lack of multiple annotators to determine inter-rater reliability metrics or ensure consensus agreement is a limitation" and that "true diagnostic interpretation ... typically demands multi-sequence and multi-planar" data. Label table (annotations / subjects): Meniscus Tear 5,658 / 663; Displaced Meniscal Tissue 232 / 56; Bone-Subchondral Edema 986 / 196; Bone Lesion 183 / 29; Bone-Fracture/Contusion/Dislocation 1,060 / 119; ACL High Grade Sprain 678 / 101; ACL Low-Mod Grade Sprain 765 / 153; MCL High Grade 11 / 4; MCL Low-Mod 285 / 121; Cartilage Full Thickness Loss/Defect 615 / 122; Cartilage Partial Thickness 2,985 / 588; Joint Effusion 1,311 / 142; Periarticular Cysts 864 / 161. GitHub license (summary of fetched page): MIT for the annotation files. https://github.com/microsoft/fastmri-plus , https://arxiv.org/abs/2109.03812

Use: annotations are MIT but need the fastMRI images (raw k-space requires reconstruction; sizes unverified). Mapping to us: effusion (142 subjects only), meniscus tear (but not medial vs lateral), cartilage defect (a weak OA proxy), edema and fracture/contusion, periarticular cysts (includes Baker's cyst, not isolated). Coronal only. Classification RISKY.

### 3.9 SKM-TEA (Stanford)

155 clinical qDESS knee scans (86 train, 33 validation, 36 test), raw k-space, DICOM, six-tissue segmentations, bounding boxes for 16 pathologies extracted from radiology reports (meniscal tears and cartilage lesions described in the dataset doc). 900 GB compressed, 1.6 TB expanded. GitHub code is MIT; the data is under the Stanford University dataset research use agreement (HF mini card: "Stanford University Dataset Research Use Agreement"). Stanford AIMI: datasets "have been available for free and are shared broadly under a license that is restricted to non-commercial purposes" (https://aimi.stanford.edu/datasets-commercial-use). Redivis account needed. Classification RISKY. Value: negligible (tiny, 3D qDESS).

### 3.10 IWOAI 2019, K2S, KNOAP2020

- IWOAI 2019: OAI subset, 176 DESS images (88 patients, 2 time points), cartilage and meniscus masks by one segmenter. "To request the dataset used for the IWOAI challenge, please contact Dr. Chaudhari." NOT USABLE.
- K2S: 300 3D fat-sat CUBE scans with 6-compartment segmentation; license "Creative Commons Attribution-NonCommercial-NoDerivatives (CC BY NC ND)"; "WE ARE NOT SHARING ANY DATA AFTER THE CHALLENGE HAS BEEN CONCLUDED." NOT USABLE.
- KNOAP2020: test set of 423 knees (female, mean age 55.7) for incident symptomatic radiographic OA; training is "OAI dataset recommended"; labels are an outcome, not our findings. NOT USABLE as external training data.

### 3.11 KMAR-50K (Sichuan, Mendeley Data)

1,190 patients, 1,444 pairs of sequences, 62,506 images; paired NIfTI with and without motion artifact; 3.0 T and 1.5 T from two machines; includes PD-TSE and PD-FSE, T1 and other routine sequences (my fetch truncated the list). Source: Sichuan Academy of Medical Sciences and Sichuan People's Hospital; Part 2 is a separate Mendeley record. License on the Mendeley page: "CC BY 4.0". Published in Scientific Data 2025 (https://www.nature.com/articles/s41597-025-05439-1). One search summary reported CC BY-NC-ND 4.0, so check the exact license on both Mendeley records before use. No pathology labels. Value: self-supervised or restoration pretraining only. Classification SAFE (conditional on the license check).

### 3.12 Other open sets

- ACL-PCL (Mendeley, East West University): CC BY 4.0, 970 augmented 640 by 640 images, four classes (normal/torn ACL, normal/torn PCL), views axial, coronal, sagittal T2 and sagittal PD-FS. Image-level labels, augmentation baked in, source hospital unclear. SAFE, negligible value.
- MeniMV (arXiv 2512.18437): 3,000 annotated exams from 750 patients over three centers, 6,000 co-registered sagittal and coronal fat-suppressed T2 scans, four-tier (0 to 3) severity grades for anterior and posterior horns, double validated by clinicians. License and download not stated in the text I could read: unverified. If released openly, the only dataset found that targets meniscus horns directly.
- KneeCoT (HF): CC-BY-NC-4.0, 25,000 3D knee MRI volumes, 1.05 TB, labels from expert reports (meniscus, ligaments, cartilage and bone marrow changes, effusion), plus VQA. Access requires users to "sign a formal ethical use agreement with the affiliated hospital prior to use"; it prohibits clinical deployment, re-identification, unauthorized redistribution and commercial applications without written authorization. Classification NOT USABLE (the page itself returned 401 on the raw README; terms are from the initial page summary).
- IEEE DataPort "Knee": segmentation of sagittal scans, 15.78 MB, "requires an IEEE DataPort Subscription". NOT USABLE.
- Chen 2018 "Knee Osteoarthritis Severity Grading Dataset" (Mendeley, CC BY 4.0, University of Florida): X-ray with KL grades, "organized from OAI". X-ray only.

### 3.13 Case collections and other modality sets

- Radiopaedia, quoted from https://radiopaedia.org/terms: prohibits "use any content obtained from any Platform to train, fine-tune, evaluate, validate or otherwise develop any machine learning or artificial intelligence model", "scrape, crawl, harvest, mirror or bulk download content from any Platform by any automated or systematic means", and states "Text and data mining of any Platform or its content is not permitted without our prior written licence". Image content is CC BY-NC-SA 3.0. NOT USABLE.
- MedPix (NLM): search summaries say CC BY-NC-SA 3.0 on the gallery and that contributed content may be copyrighted by the contributor; ML policy unverified. Case-based, unlabeled for our task. NOT USABLE in practice.
- TCIA: I found no knee osteoarthritis or knee injury collection. Knee-region collections are soft-tissue sarcoma (QIN-SARCOMA, 15 subjects, DCE-MRI; SOFT-TISSUE-SARCOMA, 51 patients), irrelevant to our 12 findings.
- MedMNIST v2: no knee MRI among its 12 2D and 6 3D datasets.
- MURA: upper-extremity X-ray only (elbow, finger, forearm, hand, humerus, shoulder, wrist), Stanford research use agreement. Not relevant.
- Kaggle knee OA X-ray sets (KL grade): X-ray only, not MRI.
- RSNA prior datasets: I found no earlier RSNA knee MRI dataset. RSNA's spine MRI and other challenges are different anatomy and carry competition-specific terms (unverified).

---

## 4. How each could be used and expected benefit

Expected benefit numbers below are my judgement; nothing here has been measured on our pipeline.

| Use mode | Candidate data | What it would do | Expected effect on weak findings |
|---|---|---|---|
| (a) Backbone pretraining | KMAR-50K (SAFE), fastMRI DICOM (RISKY, 10,000 unlabeled studies with all 5 sequences), OAI (RISKY) | Self-supervised or restoration pretraining for MRI texture and contrast variety | Small. Our backbone is already pretrained on ImageNet-style or MRI-specific models, and pretraining on unlabeled knees mostly helps when labeled data is tiny; we have about 4,400 studies. |
| (b) Mix extra labeled studies | OAI MOAKS (RISKY), fastMRI+ (RISKY), MRNet (RISKY), KneeMRI (RISKY) | Add studies for subsets of the 12 labels, mask the missing labels in the loss | OAI could matter for OA compartments, synovitis, effusion: the labels are image-graded and the lateral and patellofemoral subregions are explicit. Needs a rule to collapse subregion scores to binary, which can inject label noise. fastMRI+ adds few positives per finding and is coronal only. MRNet and KneeMRI hit findings we already do well. |
| (c) Auxiliary heads | OAI MOAKS items (cartilage, osteophyte, BML, Hoffa), fastMRI+ box labels (slice-level meniscus, cartilage, edema), 3DReasonKnee masks and boxes | Multi-task heads on shared features; fine-grained supervision can regularize | Possibly the best use of OAI: graded severity gives richer signal than binary. Same domain-shift caveats; 3D DESS must be turned into 2D-like slices or the heads trained on sagittal IW TSE FS only. |

Per weak finding:
- Lateral OA and patellofemoral OA: OAI is the only source with explicit per-compartment image-graded cartilage and osteophyte scores. Nothing SAFE helps.
- Synovitis: OAI Hoffa-synovitis and effusion-synovitis grades. fastMRI+ has no synovitis category.
- Lateral meniscus tear: MeniMV (if released) and OAI meniscus morphology; fastMRI+ meniscus boxes do not separate medial and lateral in the label table I read.

---

## 5. Risks and open questions

1. "Equally accessible" (competition rule): OAI needs NDA account plus a sponsoring institution with an FWA (DUC clause quoted above). If a participant lacks an institution, they cannot get it. This is the main reason I rate OAI RISKY rather than SAFE. Question for the hosts: does NDA-gated, free data qualify? Not answered by any source I found.
2. Winners' open-source obligation: the competition requires open-sourcing with an exemption only for "input data or pretrained models with an incompatible license". Weights trained on OAI may need an NDA Study and acknowledgement (DUC terms 11 and 12), and the "destroy all copies, versions, and data derivatives" clause (term 9) may cover weights (unverified).
3. Non-commercial terms (MRNet, SKM-TEA, KneeMRI, fastMRI, KneeCoT, OAIZIB-CM): whether a prize-awarding Kaggle competition counts as "non-commercial research" is a legal gray zone. Winners release as CC-BY-NC 4.0, which is consistent with NC inputs, but ND and "no derivative works" clauses (MRNet, KneeMRI) are riskier for trained weights. Not resolved.
4. License conflicts to settle before any use: KMAR-50K (Mendeley says CC BY 4.0, another summary says CC BY-NC-ND 4.0); OAI-ZIB original license (not found).
5. Redistributed OAI copies (HF `charwisreno/OAI-Knee-MRI`, OAIZIB-CM, the Chen X-ray set) may conflict with controlled-access terms; using them does not remove our own exposure.
6. Domain shift: OAI is 3T, 3D DESS plus 2D IW TSE, older OA-enriched population, no axial series, few acute injuries (fracture, MCL tear absent). fastMRI+ is coronal only, single reader, no per-compartment OA. 2D TSE vs DESS appearance differs (DESS shows fluid and cartilage differently).
7. Label definition mismatch: our test labels are a radiologist panel per finding; MOAKS grades are ordinal per subregion and measure different constructs (for example degenerative BML vs traumatic contusion; Hoffa-synovitis vs overall synovitis). Collapsing them needs thresholds we would have to guess, a place where "no assumptions" should be replaced by a user decision.
8. Unverified or unread: the original OAI Online DUA text (only partly read), NDA pages for OAI (JavaScript), NIA Aging Research Biobank MOST page, FNIH page (403), MeniMV release, Kaggle kneeMRI page license (only via search), fastMRI full Data Sharing Agreement text (only summarized), MedPix ML policy, raw fastMRI file sizes, OAI MCL grading, exact OAI MOAKS subset sizes (FNIH 600 verified, others unverified).
9. Compute budget: OAI is multi-TB; even a MOAKS subset of 1,000 to 2,000 knees (about 3 series each) needs download, preprocessing and likely Modal volume handling that falls under "security best practices" for controlled-access data (unverified whether Modal is compliant). That is a real cost on top of the access risk.
