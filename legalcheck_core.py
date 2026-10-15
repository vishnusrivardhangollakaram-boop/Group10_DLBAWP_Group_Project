"""
LegalCheck India - shared core logic (Group 10, ISOM5240)
=========================================================
One source of truth for the decision rule, the drafting prompt, the complaint template and the
draft quality checks. The Streamlit app imports this file, and the same code is copied verbatim
into the Colab notebooks, so the evaluated logic and the deployed logic are identical.

Pipeline 1 (fine-tuned)  : "text-classification" -> VALID / NOT_VALID legal issue in India
Pipeline 2 (pre-trained) : "text-generation"     -> body of a preliminary complaint (facts, grounds, relief)
Python                   : wraps the body in a fixed complaint template with __ [Field] placeholders
"""

# ==============================
# IMPORT PART
# ==============================
import re
import time

# ---- Settings and constants ----
VALID_LABEL = "VALID"
NOT_VALID_LABEL = "NOT_VALID"
VALIDITY_THRESHOLD = 0.50            # VALID when the model score for VALID is at least this value
MINIMUM_INPUT_WORDS = 8              # shorter inputs are rejected with a helpful message
MAXIMUM_INPUT_WORDS = 350            # longer inputs are rejected (no silent truncation of facts)
CLASSIFIER_MAX_TOKENS = 128          # same max_length as used during fine-tuning
DRAFT_MAX_NEW_TOKENS = 420
DRAFT_REPETITION_PENALTY = 1.10
FACT_RETENTION_PASS_LEVEL = 0.75     # share of key facts that must appear in the draft
DRAFT_MINIMUM_WORDS = 60
DRAFT_MAXIMUM_WORDS = 420

DRAFT_SYSTEM_PROMPT = (
    "You are a careful legal drafting assistant for an Indian law firm. "
    "Use only the facts the client gave. Never invent names, dates, amounts, places, documents, witnesses "
    "or section numbers. If a needed detail is missing, write a placeholder such as __ [Date of incident]. "
    "Write in formal English."
)

DRAFT_USER_TEMPLATE = (
    "Client narrative:\n\"\"\"{narrative}\"\"\"\n\n"
    "Write the body of a preliminary complaint using exactly these three headings:\n"
    "FACTS OF THE CASE: numbered paragraphs (1., 2., 3., ...) restating the facts in the third person "
    "(\"The complainant ...\"), keeping every name, date, place and amount given.\n"
    "GROUNDS: two or three numbered points explaining briefly why this is a legal grievance and its nature "
    "(criminal, civil, consumer, family, labour or cyber).\n"
    "RELIEF SOUGHT: one to three numbered requests.\n"
    "Do not cite any section numbers or case law. Keep it under 300 words."
)


# ==============================
# FUNCTION PART
# ==============================
def count_words(text):
    """Return the number of words in a text."""
    return len(re.findall(pattern=r"\S+", string=text or ""))


def validate_narrative(narrative):
    """Check the user's narrative. Returns (is_ok, message)."""
    word_count = count_words(text=narrative)
    if word_count < MINIMUM_INPUT_WORDS:
        return False, f"Please describe the issue in at least {MINIMUM_INPUT_WORDS} words (you wrote {word_count})."
    if word_count > MAXIMUM_INPUT_WORDS:
        return False, (f"Please shorten the description to {MAXIMUM_INPUT_WORDS} words or fewer "
                       f"(you wrote {word_count}); facts are never cut silently.")
    return True, "ok"


def decide_validity(classifier_output):
    """Turn text-classification output (list of {label, score}) into (decision_label, valid_score)."""
    scores = {item["label"]: float(item["score"]) for item in classifier_output}
    valid_score = scores.get(VALID_LABEL, 1.0 - scores.get(NOT_VALID_LABEL, 0.0))
    decision_label = VALID_LABEL if valid_score >= VALIDITY_THRESHOLD else NOT_VALID_LABEL
    return decision_label, valid_score


def classify_narrative(classifier, narrative):
    """Run Pipeline 1 on one narrative. Returns (decision_label, valid_score, seconds)."""
    start_time = time.perf_counter()
    classifier_output = classifier(narrative, top_k=None, truncation=True, max_length=CLASSIFIER_MAX_TOKENS)
    if classifier_output and isinstance(classifier_output[0], list):      # some versions nest the list
        classifier_output = classifier_output[0]
    decision_label, valid_score = decide_validity(classifier_output=classifier_output)
    return decision_label, valid_score, time.perf_counter() - start_time


def build_draft_prompt(tokenizer, narrative):
    """Build the chat prompt string for Pipeline 2 (thinking disabled for Qwen3; ignored by other models)."""
    messages = [
        {"role": "system", "content": DRAFT_SYSTEM_PROMPT},
        {"role": "user", "content": DRAFT_USER_TEMPLATE.format(narrative=narrative.strip())},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)


def clean_generated_body(raw_text):
    """Remove empty thinking blocks, markdown symbols and extra blank lines from the generated body."""
    text = re.sub(pattern=r"<think>.*?</think>", repl="", string=raw_text or "", flags=re.DOTALL)
    text = text.replace("**", "").replace("##", "").replace("```", "")
    text = re.sub(pattern=r"\n{3,}", repl="\n\n", string=text)
    return text.strip()


def generate_draft_body(generator, narrative, streamer=None):
    """Run Pipeline 2 on one narrative. Returns (body_text, seconds)."""
    prompt = build_draft_prompt(tokenizer=generator.tokenizer, narrative=narrative)
    start_time = time.perf_counter()
    generation_kwargs = dict(max_new_tokens=DRAFT_MAX_NEW_TOKENS, do_sample=False,
                             repetition_penalty=DRAFT_REPETITION_PENALTY, return_full_text=False)
    if streamer is not None:
        generation_kwargs["streamer"] = streamer
    output = generator(prompt, **generation_kwargs)
    body_text = clean_generated_body(raw_text=output[0]["generated_text"])
    return body_text, time.perf_counter() - start_time


def assemble_complaint(body_text, narrative_subject="__ [Short subject of the complaint]"):
    """Wrap the generated body in the fixed preliminary-complaint template (Python controls the layout)."""
    header = (
        "PRELIMINARY COMPLAINT (DRAFT FOR ADVOCATE REVIEW)\n\n"
        "BEFORE: __ [Court / Police Station / Consumer Forum name], __ [Place]\n\n"
        "COMPLAINANT: __ [Complainant full name], __ [Age], __ [Address], __ [Phone]\n"
        "AGAINST: __ [Respondent / accused name], __ [Address]\n\n"
        f"SUBJECT: {narrative_subject}\n\n"
        "Respected Sir / Madam,\n\n"
    )
    footer = (
        "\n\nLIST OF DOCUMENTS: __ [Documents to be attached, e.g. receipts, messages, photos]\n\n"
        "VERIFICATION: I, __ [Complainant full name], state that the facts above are true to the best of my "
        "knowledge and belief.\n\n"
        "Place: __ [Place]        Date: __ [Date]        Signature: __ [Signature]\n\n"
        "Note: Machine-generated preliminary draft. It must be reviewed and completed by an advocate."
    )
    return f"{header}{body_text.strip()}{footer}"


def list_missing_fields(draft_text):
    """Return the unique __ [Field] placeholders still present in a draft (missing-information checklist)."""
    fields = re.findall(pattern=r"__ \[([^\]]+)\]", string=draft_text or "")
    return list(dict.fromkeys(field.strip() for field in fields))


def score_draft_body(body_text, key_facts):
    """Objective rubric for one generated body. Returns a dict of checks and an overall pass flag.

    Checks: (1) key-fact retention >= 75 %, (2) required headings + numbered paragraphs,
            (3) no invented section numbers / case law, (4) clean length and no stray reasoning text.
    """
    lower_body = (body_text or "").lower()
    facts = [fact.strip() for fact in key_facts if fact.strip()]
    retained = sum(1 for fact in facts if fact.lower() in lower_body)
    fact_retention = retained / len(facts) if facts else 1.0
    has_structure = ("facts" in lower_body and "relief" in lower_body
                     and len(re.findall(pattern=r"(?m)^\s*\d+[.)]", string=body_text or "")) >= 2)
    no_invented_law = re.search(pattern=r"\b(section|sec\.|article|u/s)\s*\d+|\bipc\b|\bbns\b|\bcrpc\b", string=lower_body) is None
    word_total = count_words(text=body_text)
    clean_length = (DRAFT_MINIMUM_WORDS <= word_total <= DRAFT_MAXIMUM_WORDS and "<think" not in lower_body
                    and re.search(pattern=r"[一-鿿]", string=body_text or "") is None)
    checks = {
        "fact_retention": round(fact_retention, 2),
        "facts_ok": fact_retention >= FACT_RETENTION_PASS_LEVEL,
        "structure_ok": has_structure,
        "no_invented_law_ok": no_invented_law,
        "clean_length_ok": clean_length,
        "words": word_total,
    }
    checks["pass"] = all([checks["facts_ok"], checks["structure_ok"], checks["no_invented_law_ok"],
                          checks["clean_length_ok"]])
    return checks
