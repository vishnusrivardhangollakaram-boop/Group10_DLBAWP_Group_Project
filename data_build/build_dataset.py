"""
LegalCheck India - dataset builder (Group 10, ISOM5240)
=======================================================
Builds an English, client-style dataset for the fine-tuned pipeline:
    label 1 = VALID   : the narrative describes a legally actionable dispute / issue in India
                        (criminal, civil, consumer, family, labour, property or cyber matter).
    label 0 = NOT_VALID: the narrative is a social, emotional or matter-of-taste grievance, a lawful act,
                        an unenforceable claim (e.g. a wager), a purely hypothetical worry, or a matter
                        with no Indian connection.

Leakage control: each narrative "family" (scenario template) is assigned to ONE split only.
Training, validation and test each use DIFFERENT families, so validation (used for model and epoch
selection) and test (final check) both measure generalisation to new kinds of narratives,
not memorised wording.

Run:  python build_dataset.py   (writes CSV files to ../data)
"""

# ==============================
# IMPORT PART
# ==============================
import csv
import hashlib
import json
import os
import random

RANDOM_SEED = 5240
OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
SAMPLES_PER_FAMILY = {"train": 212, "validation": 126, "test": 84}   # unique variations per family, by split
# -> about 5,000 training, 1,000 validation and 1,000 test rows, each balanced 50/50
TEST_FAMILY_SHARE = 0.25          # share of families held out completely for the test set
VALIDATION_FAMILY_SHARE = 0.20    # share of families held out completely for validation (model/epoch selection)

# ---- Slot values (Indian context) ----
NAMES = ["Ravi", "Priya", "Anil", "Sunita", "Mohammed", "Lakshmi", "Arjun", "Fatima", "Suresh", "Meena",
         "Vikram", "Kavya", "Rahul", "Deepa", "Harish", "Anjali", "Imran", "Pooja", "Karthik", "Neha",
         "Gopal", "Sneha", "Ramesh", "Divya", "Joseph", "Asha", "Naveen", "Rekha", "Sanjay", "Farah"]
CITIES = ["Hyderabad", "Vijayawada", "Chennai", "Bengaluru", "Mumbai", "Pune", "Delhi", "Kolkata", "Lucknow",
          "Jaipur", "Kochi", "Visakhapatnam", "Guntur", "Nagpur", "Indore", "Bhopal", "Patna", "Ahmedabad"]
FOREIGN_CITIES = ["London", "Dubai", "New York", "Singapore", "Toronto", "Sydney", "Berlin", "Tokyo"]
AMOUNTS = ["5,000", "12,500", "25,000", "40,000", "75,000", "1,20,000", "2,50,000", "3,00,000", "8,000", "60,000"]
DATES = ["last Monday", "on 3 March 2025", "two weeks ago", "in January", "on 14 August 2025", "last month",
         "on 22 June 2025", "three days ago", "in December last year", "on 9 September 2025"]
RELATIONS = ["neighbour", "cousin", "colleague", "friend", "classmate", "brother-in-law", "flatmate"]
PLATFORMS = ["Instagram", "WhatsApp", "Facebook", "Telegram", "X (Twitter)"]
OPENERS = ["", "Sir, ", "Hello, ", "Please help. ", "I need advice. ", "Madam, ", "Good morning. "]
CLOSERS = ["", " What can I do?", " Can I take legal action?", " Please guide me.", " Is this a case?",
           " Should I go to court?", " I want justice.", " Is there any remedy?"]

# ---- VALID families (label 1) ----
VALID_FAMILIES = {
    "upi_fraud": [
        "Someone called me {date} pretending to be from my bank and took my OTP. Rs {amount} was debited from my account in {city} without my consent.",
        "A caller claiming to be a bank officer made me share an OTP {date} and Rs {amount} vanished from my savings account. I live in {city}.",
        "I got a fake KYC update link {date}, entered my details and Rs {amount} was transferred out of my account by unknown people in {city}."],
    "online_shopping_fraud": [
        "I paid Rs {amount} to an online seller on {platform} for a phone {date}. He blocked me and never delivered anything.",
        "A seller on {platform} took Rs {amount} as advance for furniture {date} and stopped responding. No product came to my house in {city}."],
    "cheque_bounce": [
        "My {relation} {name} gave me a cheque of Rs {amount} to repay a loan and it bounced {date} for insufficient funds. He refuses to pay.",
        "I lent Rs {amount} to {name} in {city}. The cheque he issued to repay me was dishonoured by the bank {date}."],
    "deposit_not_returned": [
        "I vacated my rented flat in {city} {date}. The landlord is refusing to return my security deposit of Rs {amount} though I left the flat in good condition.",
        "My landlord {name} kept my Rs {amount} rental deposit after I moved out {date} and gives no reason."],
    "salary_unpaid": [
        "My employer in {city} has not paid my salary of Rs {amount} for the last three months and stopped answering my calls.",
        "I resigned {date} after serving notice, but the company has withheld my final settlement of Rs {amount}."],
    "defective_product": [
        "I bought a refrigerator for Rs {amount} {date} from a showroom in {city}. It stopped working in a week and the dealer refuses repair or refund even under warranty.",
        "The washing machine I purchased {date} was defective from day one. The company in {city} is refusing replacement despite warranty."],
    "assault": [
        "My {relation} {name} beat me with a stick {date} during an argument in {city}. I was injured and went to hospital.",
        "{name} slapped and punched me in public {date} near my house in {city}. I have injury marks."],
    "theft": [
        "My motorbike was stolen from outside my house in {city} {date}. I suspect my {relation} {name} who has a duplicate key.",
        "Someone broke into my shop in {city} {date} and stole cash of Rs {amount} and two laptops."],
    "dowry_harassment": [
        "My husband and in-laws in {city} keep demanding Rs {amount} more as dowry and beat me when I refuse.",
        "Since marriage my in-laws harass me for dowry. {date} they threw me out of the house for not bringing Rs {amount}."],
    "cyber_harassment": [
        "Someone created a fake {platform} account in my name and is posting my photos with obscene comments. It started {date}.",
        "A man I do not know keeps sending me threatening and vulgar messages on {platform} since {date} and posted my morphed photos."],
    "criminal_intimidation": [
        "My {relation} {name} threatened to kill me and my family {date} if I do not sell him my land in {city}.",
        "Loan recovery agents came to my house in {city} {date}, abused my parents and threatened to harm us."],
    "encroachment": [
        "My {relation} {name} has built a wall on part of my plot in {city} {date} and refuses to remove it. I have the sale deed.",
        "{name} has occupied my agricultural land near {city} without permission and is cultivating it."],
    "builder_delay": [
        "I paid Rs {amount} to a builder in {city} for a flat in 2021. Possession was promised in 2023 but still not given and they refuse a refund.",
        "The builder took my full payment for an apartment in {city} but has not handed over possession for three years."],
    "medical_negligence": [
        "During a simple surgery at a private hospital in {city} {date}, the doctor left a cotton swab inside my mother. She had to be operated again.",
        "A hospital in {city} gave my father the wrong injection {date} and his condition became critical. They deny responsibility."],
    "defamation": [
        "{name} has been spreading false stories that I stole money from the temple fund, and posted it on {platform}. My reputation in {city} is damaged.",
        "My former business partner published false allegations about me in a local newspaper in {city} {date}."],
    "road_accident": [
        "A car hit my scooter from behind in {city} {date} and the driver ran away. I fractured my leg. I noted the number plate.",
        "A rash bus driver hit my son on the road in {city} {date}. He was badly hurt."],
    "forgery": [
        "My {relation} {name} forged my signature on property papers and sold my house in {city} without my knowledge.",
        "Someone forged my signature to take a loan of Rs {amount} in my name from a finance company in {city}."],
    "workplace_harassment": [
        "My manager in {city} keeps touching me inappropriately and making sexual comments. When I objected he threatened my job.",
        "A senior colleague sexually harassed me at the office {date}. HR in {city} ignored my complaint."],
    "maintenance_refused": [
        "My husband left me and our two children {date} and refuses to give any money for food and school fees. He earns well in {city}.",
        "After separation my husband stopped supporting our child. He has a good job in {city} but gives nothing."],
    "insurance_rejected": [
        "My health insurer rejected my hospital claim of Rs {amount} saying the illness was pre-existing, though I disclosed everything when buying the policy.",
        "The insurance company is refusing to pay for my car accident damage of Rs {amount} without a valid reason."],
    "trespass_damage": [
        "My {relation} {name} entered my house in {city} without permission {date} and broke my furniture.",
        "Some people entered my farm near {city} {date}, cut my mango trees and took the wood."],
    "investment_scam": [
        "A company in {city} promised 10 percent monthly returns. I invested Rs {amount} {date} and now their office is closed and phones are off.",
        "{name} took Rs {amount} from me promising a government job and now refuses to return the money."],
}

# ---- NOT VALID families (label 0) ----
NOT_VALID_FAMILIES = {
    "not_invited": [
        "My {relation} {name} did not invite me to his wedding in {city} {date}. I feel insulted.",
        "My best friend {name} had a birthday party {date} and did not invite me. I want to file a case."],
    "breakup": [
        "My girlfriend broke up with me {date} because she likes someone else. I want to take action against her.",
        "{name} ended our relationship {date} and blocked me on {platform}. Can I sue her for hurting my feelings?"],
    "food_taste": [
        "I ordered biryani at a restaurant in {city} {date}. It was served as ordered and I paid, but it tasted bland to me.",
        "The coffee at a cafe in {city} was too sweet for my liking. They served exactly what I ordered."],
    "boring_movie": [
        "I watched a movie in a theatre in {city} {date} and the story was very boring. I want my ticket money back from the actors.",
        "The cricket team lost the match {date} and I am very upset. I want to sue the players for disappointing fans."],
    "unfollowed": [
        "My {relation} {name} unfollowed me on {platform} {date}. I think this is disrespectful.",
        "{name} left our {platform} group without saying goodbye. I feel bad about it."],
    "appraisal": [
        "My manager in {city} gave me a 'meets expectations' rating instead of 'exceeds'. I got my full salary and bonus as per policy, but I expected more.",
        "I was not selected for promotion this year though my salary and all dues were paid on time. My colleague got promoted instead."],
    "paint_colour": [
        "My {relation} {name} painted his own house bright pink {date}. I do not like the colour. Can I stop him?",
        "The neighbour in {city} planted flowers in his own garden that I find ugly."],
    "lost_bet": [
        "I lost a bet of Rs {amount} with {name} on a cricket match. I want the court to force him to return my bet money.",
        "{name} won Rs {amount} from me in a bet on the elections. Now I want to recover it legally."],
    "foreign_matter": [
        "I am a British citizen living in {foreign}. My British neighbour there played loud music {date}. Can an Indian court act?",
        "Two of my friends, both foreign nationals, argued over a parking spot in {foreign} {date}. Nothing happened in India."],
    "hypothetical": [
        "I feel that my {relation} {name} might cheat me someday in future, though he has done nothing yet.",
        "What if my landlord increases the rent next year? Nothing has happened yet, I am just worried."],
    "discount_refused": [
        "A shop in {city} refused to give me a discount on a shirt {date}. The price was clearly marked and I chose to pay it.",
        "The auto driver in {city} charged the meter fare {date}. I wanted him to reduce it but he refused."],
    "weather_forecast": [
        "The weather app said no rain {date} but it rained and my clothes got wet in {city}. I want to sue the app.",
        "The weather department predicted sunshine but it rained during my picnic near {city}."],
    "rude_sibling": [
        "My younger brother speaks rudely to me at home and does not respect my opinions.",
        "My {relation} {name} ignored my advice about his career {date}. I feel he does not value me."],
    "lawful_bank_charge": [
        "My bank deducted a Rs 50 SMS charge that is listed in the account terms I signed. I just do not like paying it.",
        "The gym in {city} did not refund my fee after I stopped going, and the membership form I signed clearly said no refunds after 30 days of use."],
    "game_loss": [
        "I lost a chess tournament in {city} {date} fairly. I think the winner was just lucky.",
        "My team lost a friendly football match {date}. The referee's decisions were correct but I am still angry."],
    "general_question": [
        "How do I cook a good biryani at home for guests?",
        "Which is the best tourist place to visit in {city} in winter?",
        "Can you suggest a good laptop under Rs {amount} for college?"],
    "late_reply": [
        "My {relation} {name} took two days to reply to my message on {platform}. I think he is avoiding me.",
        "{name} did not wish me on my birthday {date}. I feel hurt."],
    "own_mistake": [
        "I forgot my umbrella in a bus in {city} {date} and nobody returned it. I did not report it to anyone.",
        "I accidentally deleted my own photos from my phone {date}. Who can I blame legally?"],
    "dream_complaint": [
        "I dreamt that my {relation} {name} stole my car. In reality nothing happened, but I want to complain.",
        "In a dream I saw my boss firing me. It did not actually happen."],
    "celebrity_opinion": [
        "A film star said in an interview that he prefers tea over coffee. I love coffee and feel offended.",
        "A famous singer did not reply to my fan letter. I want action against her."],
    "sports_selection": [
        "My son was not selected for the school cricket team after a fair trial in {city}. Other boys simply performed better.",
        "I was not chosen as class monitor though the teacher held a fair vote in {city}."],
    "traffic_jam": [
        "I was stuck in a traffic jam in {city} for two hours {date} and missed a movie.",
        "The train from {city} was ten minutes late {date} and I was bored waiting."],
}


# ==============================
# FUNCTION PART
# ==============================
def fill_template(template, random_generator):
    """Fill one narrative template with random Indian-context slot values and an opener/closer."""
    slot_values = {
        "name": random_generator.choice(NAMES),
        "city": random_generator.choice(CITIES),
        "foreign": random_generator.choice(FOREIGN_CITIES),
        "amount": random_generator.choice(AMOUNTS),
        "date": random_generator.choice(DATES),
        "relation": random_generator.choice(RELATIONS),
        "platform": random_generator.choice(PLATFORMS),
    }
    body = template.format(**slot_values)
    opener = random_generator.choice(OPENERS)
    closer = random_generator.choice(CLOSERS)
    return f"{opener}{body}{closer}".strip()


def generate_family_rows(family_name, templates, label, sample_count, random_generator):
    """Generate unique narratives for one scenario family."""
    rows, seen_texts = [], set()
    attempts = 0
    while len(rows) < sample_count and attempts < sample_count * 50:
        attempts += 1
        text = fill_template(template=random_generator.choice(templates), random_generator=random_generator)
        if text in seen_texts:
            continue
        seen_texts.add(text)
        rows.append({"text": text, "label": label, "family": family_name})
    return rows


def assign_families(family_names, random_generator):
    """Assign every family to exactly one split: returns {family_name: 'train' | 'validation' | 'test'}."""
    shuffled = sorted(family_names)
    random_generator.shuffle(shuffled)
    test_count = max(1, round(len(shuffled) * TEST_FAMILY_SHARE))
    validation_count = max(1, round(len(shuffled) * VALIDATION_FAMILY_SHARE))
    assignment = {}
    for index, family_name in enumerate(shuffled):
        if index < test_count:
            assignment[family_name] = "test"
        elif index < test_count + validation_count:
            assignment[family_name] = "validation"
        else:
            assignment[family_name] = "train"
    return assignment


def write_csv(rows, file_name):
    """Write rows (dicts) to data/<file_name> with a stable column order and a sample id."""
    path = os.path.join(OUTPUT_FOLDER, file_name)
    with open(path, "w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=["sample_id", "text", "label", "label_name", "family"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def add_ids(rows, prefix):
    """Add sample_id and readable label_name to every row."""
    for index, row in enumerate(rows):
        row["sample_id"] = f"{prefix}_{index:04d}"
        row["label_name"] = "VALID" if row["label"] == 1 else "NOT_VALID"
    return rows


def file_sha256(path):
    """Return the SHA-256 hash of a file (for the split manifest)."""
    with open(path, "rb") as file_handle:
        return hashlib.sha256(file_handle.read()).hexdigest()


def main():
    """Build train / validation / test CSV files and a manifest."""
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    random_generator = random.Random(RANDOM_SEED)

    family_split = {}
    family_split.update(assign_families(family_names=VALID_FAMILIES.keys(), random_generator=random_generator))
    family_split.update(assign_families(family_names=NOT_VALID_FAMILIES.keys(), random_generator=random_generator))

    split_rows = {"train": [], "validation": [], "test": []}
    for families, label in [(VALID_FAMILIES, 1), (NOT_VALID_FAMILIES, 0)]:
        for family_name, templates in families.items():
            rows = generate_family_rows(family_name=family_name, templates=templates, label=label,
                                        sample_count=SAMPLES_PER_FAMILY[family_split[family_name]],
                                        random_generator=random_generator)
            split_rows[family_split[family_name]].extend(rows)
    for rows in split_rows.values():
        random_generator.shuffle(rows)
    train_rows, validation_rows, test_rows = split_rows["train"], split_rows["validation"], split_rows["test"]

    manifest = {"seed": RANDOM_SEED, "family_split": dict(sorted(family_split.items())), "files": {}}
    for rows, prefix, file_name in [(train_rows, "train", "train.csv"), (validation_rows, "val", "validation.csv"),
                                    (test_rows, "test", "test.csv")]:
        path = write_csv(rows=add_ids(rows=rows, prefix=prefix), file_name=file_name)
        positives = sum(row["label"] for row in rows)
        manifest["files"][file_name] = {"rows": len(rows), "valid": positives, "not_valid": len(rows) - positives,
                                        "sha256": file_sha256(path=path)}
        print(f"{file_name:16s} rows={len(rows):5d}  valid={positives:5d}  not_valid={len(rows) - positives:5d}")

    with open(os.path.join(OUTPUT_FOLDER, "split_manifest.json"), "w", encoding="utf-8") as file_handle:
        json.dump(manifest, file_handle, indent=2)
    for split_name in ["validation", "test"]:
        print(f"{split_name} families: {sorted(name for name, split in family_split.items() if split == split_name)}")


if __name__ == "__main__":
    main()
