# 🩻 Medical: Spine CT Segmentation Fairness
**Target**: bias evaluation of 3D segmentation models.

## Overview
This scenario evaluates a pre-trained **Spine CT Segmentation** model (from Hugging Face) on the **SPINE-METS-CT-SEG** dataset. It specifically audits the model for performance disparities across demographic groups (Age, Sex).

## Data
We leverage the data downloaded in `samples-extra/scenarios/surgery-dicom-tcia`.
**Note**: You must have run the downloader in `samples-extra` first!

## Workflow
1. **01_model_evaluation.py**: Loads the HF model and runs inference on the local dataset.
2. **02_fairness_audit.py**: Audits the segmentation metrics (Dice Score) for bias.

## Requirements
- `venturalitica` SDK
- `monai` for medical deep learning
- `torch`
