# ⚖️ LegalCheck India: Legal-Issue Check and Preliminary Complaint Draft

**ISOM5240 Deep Learning Business Applications with Python, Group 10**
Gollakaram Vishnu Srivardhan (21401910, group leader) · Rahul Singh (2610gnam02)
Proposed for **Karavadi & Associates** ([karavadi.in](https://karavadi.in/)). This is a non-commercial coursework prototype and not legal advice.

The user describes a problem in plain English. The app then:

1. **Pipeline 1 (fine-tuned, `text-classification`)** decides whether the narrative is a **legally valid dispute or issue in India**.
2. **NOT VALID:** the app says so, shows the model score and offers **New query**.
3. **VALID:** **Pipeline 2 (pre-trained, `text-generation`)** writes the facts, grounds and relief of a **preliminary complaint**. Python wraps them in a fixed template with `__ [Field]` placeholders. The draft is editable on screen, with a missing-information checklist, and downloads as an editable **Word (.docx)** file.

| Pipeline | Model | Candidates compared |
|---|---|---|
| 1 · validity (fine-tuned) | [`VishnuGollakaram/legalcheck-india-validity`](https://huggingface.co/VishnuGollakaram/legalcheck-india-validity) | distilbert-base-uncased, bert-base-uncased, law-ai/InLegalBERT |
| 2 · drafting (pre-trained) | `Qwen/Qwen3-0.6B` (thinking disabled) | Qwen2.5-0.5B-Instruct, Qwen3-0.6B, SmolLM2-360M-Instruct |

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit interface. It follows the course structure of an IMPORT part, a FUNCTION part and a MAIN part. |
| `legalcheck_core.py` | Shared logic: input checks, decision rule, prompt, complaint template, missing fields and the draft rubric. The notebooks contain a verbatim copy of it. |
| `requirements.txt` | Pinned version ranges for Streamlit Cloud (CPU). |
| `.streamlit/config.toml` | Light theme. |
| `tests/run_offline_tests.py` | 15 offline tests that use fake pipelines and need no internet: `python tests/run_offline_tests.py`. |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Behaviour and safeguards

- Inputs shorter than 8 words or longer than 350 words are rejected with a message. Facts are never silently truncated.
- If a model fails, the app shows a visible error. It never falls back to a blank template without saying so.
- Editing the narrative clears the old verdict. The download always uses the current edited text.
- Drafts never cite section numbers. Unknown details stay as `__ [Field]` placeholders for the advocate to complete.

Acknowledgement: generative AI (Claude) was used as a coding assistant, as the course syllabus permits.
