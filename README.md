# CaseScreen India — Group 10

ISOM5240 Deep Learning Business Applications with Python. Gollakaram Vishnu Srivardhan (21401910), group leader; Rahul Singh (2610gnam02).

An educational legal-intake prototype for India: screen an account, request missing details, and prepare an editable factual complaint for advocate review when a potential issue is identified. It does not determine legal validity or predict applicable statutory sections.

## Current status

Implementation and evaluation are in progress. The dataset audit and three pretrained screening-model comparisons are complete. The selected MiniLM model is being fine-tuned. Generator comparison, full integration verification and public deployment remain pending. This repository is not yet a completed submission.

## Two model pipelines

1. **Fine-tuned topic classifier.** Compare three pretrained NLI base models on the same 400 validation examples. Select by accuracy, then warm inference runtime. Fine-tune only the winner on 5,000 training examples. Choose the best epoch on validation accuracy and evaluate once on 400 held-out test examples.
2. **Unchanged pretrained generator.** Compare three Qwen instruction models on identical fictional intake scenarios. Select by strict workflow accuracy (category plus valid, source-grounded output), then runtime. No generator fine-tuning. Python arranges the selected factual quotations into an editable complaint template.

Dataset topic labels are weak labels derived from source membership. They do not establish whether an Indian legal claim is valid. The synthetic scenario benchmark has not been adjudicated by legal professionals.

## Run

Use Python 3.11 or 3.12 and install `requirements.txt`. Run `streamlit run app.py` from this directory after the selected result files and fine-tuned model are available. If using a published checkpoint, set `SCREENING_MODEL_ID` to its actual Hugging Face repository ID. No placeholder model ID is provided.

The app rejects unsupported generated facts, retains the full original account in the draft, uses explicit placeholders for unknown details and exports the current edited text to Word. Changing input clears prior results. New query clears the active case widgets.

## Experiments

Install `requirements-training.txt`. Execute notebooks in order: `01_Data_Audit.ipynb`, `02_Classifier_Training.ipynb`, `03_Generation_Comparison.ipynb`. The supplied frozen processed splits can be used directly for training. Full source rebuilding requires the pinned source data documented in `data/processed/audit.json`.

Notebook outputs are actual recorded execution results. The managed authoring host cannot start a socket-based Jupyter kernel, so plain Python cells are executed in order in a fresh Python process with captured output. A separate Colab Restart-and-Run-All check remains outstanding.

All source revisions are pinned in `model_manifest.json`. Runtime is measured on CPU with a separate load phase and warm-up. Failures stay in evaluation denominators. Public Streamlit evaluation must be recorded separately from local results.

## Data attribution

- [Indian-legal-data-v3](https://huggingface.co/datasets/kaushik-harsh-99/Indian-legal-data-v3), MIT. Instruction text only; supplied legal answers are not used as legal authority.
- [CLINC150](https://huggingface.co/datasets/clinc/clinc_oos), CC BY 3.0. Restricted everyday-intent subset. See audit for pinned revisions and transformations.

All development scenarios are fictional and AI-authored. Use anonymised or fictional data in the public classroom demo.
