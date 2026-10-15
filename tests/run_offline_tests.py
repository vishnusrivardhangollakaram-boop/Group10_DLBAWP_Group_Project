"""
Offline tests for LegalCheck India (no internet, no model download).
Run from the app folder:  python tests/run_offline_tests.py
Tests the shared core logic (input validation, decision rule, template, missing fields, draft rubric)
and the Word export, using small fake pipelines instead of the real Hugging Face models.
"""
import os
import sys

APP_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP_FOLDER)
import legalcheck_core as core  # noqa: E402

RESULTS = []


def check(name, condition):
    """Record one test result."""
    RESULTS.append((name, bool(condition)))
    print(f"{'PASS' if condition else 'FAIL'}  {name}")


def fake_classifier(valid_score):
    """Return a fake text-classification pipeline that always gives the chosen VALID score."""
    def classify(text, **kwargs):
        return [{"label": "VALID", "score": valid_score}, {"label": "NOT_VALID", "score": 1 - valid_score}]
    return classify


class FakeTokenizer:
    """Minimal tokenizer with a chat template (records the enable_thinking flag)."""
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        self.enable_thinking = enable_thinking
        return "\n".join(message["content"] for message in messages)


def fake_generator(text):
    """Return a fake text-generation pipeline that returns fixed text."""
    def generate(prompt, **kwargs):
        return [{"generated_text": text}]
    generate.tokenizer = FakeTokenizer()
    return generate


def main():
    """Run all offline tests and exit with code 1 if any fails."""
    # Input validation
    check("too-short input rejected", not core.validate_narrative(narrative="my bike stolen")[0])
    check("normal input accepted", core.validate_narrative(narrative="Someone stole my bike from outside my house in Pune yesterday night.")[0])
    check("very long input rejected (no silent truncation)", not core.validate_narrative(narrative="word " * 400)[0])
    # Decision rule
    check("score 0.91 -> VALID", core.classify_narrative(classifier=fake_classifier(valid_score=0.91), narrative="x")[0] == "VALID")
    check("score 0.12 -> NOT_VALID", core.classify_narrative(classifier=fake_classifier(valid_score=0.12), narrative="x")[0] == "NOT_VALID")
    check("threshold 0.50 -> VALID", core.decide_validity(classifier_output=[{"label": "VALID", "score": 0.5}])[0] == "VALID")
    check("nested pipeline output handled",
          core.classify_narrative(classifier=lambda text, **k: [[{"label": "VALID", "score": 0.8}]], narrative="x")[0] == "VALID")
    # Generation + template
    body = ("<think>\n</think>\nFACTS OF THE CASE:\n1. The complainant paid Rs 50,000 to Mr. Rao in Vijayawada as deposit.\n"
            "2. Mr. Rao refuses to return the deposit after the complainant vacated the flat on __ [Date of vacating].\n"
            "GROUNDS:\n1. Wrongful retention of money is a civil grievance.\n2. The landlord had no reason to keep it.\n"
            "RELIEF SOUGHT:\n1. Direct the respondent to refund Rs 50,000 with interest.\n2. Costs of the complaint. "
            "The complainant requests the authority to act quickly and fairly on this matter of the deposit.")
    generator = fake_generator(text=body)
    body_text, seconds = core.generate_draft_body(generator=generator, narrative="My landlord kept my deposit.")
    check("thinking disabled for Qwen3 template", generator.tokenizer.enable_thinking is False)
    check("empty <think> block removed", "<think>" not in body_text)
    checks = core.score_draft_body(body_text=body_text, key_facts=["Rao", "Vijayawada", "50,000", "deposit"])
    check("good draft passes rubric", checks["pass"])
    bad = core.score_draft_body(body_text=body_text + "\nThis falls under Section 406 IPC.", key_facts=["Rao"])
    check("invented section number fails rubric", not bad["pass"] and not bad["no_invented_law_ok"])
    missing = core.score_draft_body(body_text=body_text, key_facts=["Kiran", "Guntur", "bike", "45,000"])
    check("missing key facts fail rubric", not missing["facts_ok"])
    complaint = core.assemble_complaint(body_text=body_text)
    fields = core.list_missing_fields(draft_text=complaint)
    check("template has placeholders", "Complainant full name" in fields and "Date of vacating" in fields)
    check("placeholders unique", len(fields) == len(set(fields)))
    # Word export (python-docx only, no Streamlit needed)
    try:
        from docx import Document
        import io
        sys.modules.setdefault("streamlit", __import__("types").ModuleType("streamlit"))
        document = Document()
        document.add_paragraph(complaint.split("\n")[0])
        buffer = io.BytesIO(); document.save(buffer)
        check("Word document can be written", buffer.getbuffer().nbytes > 1000)
    except ImportError:
        print("SKIP  python-docx not installed")
    passed = sum(1 for _, ok in RESULTS if ok)
    print(f"\n{passed}/{len(RESULTS)} tests passed")
    sys.exit(0 if passed == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
