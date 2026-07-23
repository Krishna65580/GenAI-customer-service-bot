"""
Parses the real MedQuAD dataset (https://github.com/abachaa/MedQuAD) XML
files into a flat CSV of question/answer pairs for retrieval indexing.

Run once to (re)build data/medquad.csv from the cloned data/MedQuAD_repo/
source tree. This is a one-time offline preprocessing step, not something
the Streamlit app does at runtime.
"""

import os
import csv
import xml.etree.ElementTree as ET

REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "MedQuAD_repo")
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "medquad.csv")

# Cap per source-folder and overall, to keep the demo index a manageable
# size (the full MedQuAD dataset has 47,000+ QA pairs across 12 sources).
MAX_PER_FOLDER = 400
MAX_TOTAL = 3000


def parse_file(path):
    try:
        tree = ET.parse(path)
    except ET.ParseError:
        return []
    root = tree.getroot()
    focus = root.findtext("Focus", default="").strip()
    source = root.attrib.get("source", "")
    rows = []
    for qapair in root.iter("QAPair"):
        q_el = qapair.find("Question")
        a_el = qapair.find("Answer")
        if q_el is None or a_el is None:
            continue
        question = (q_el.text or "").strip()
        answer = (a_el.text or "").strip()
        qtype = q_el.attrib.get("qtype", "")
        if not question or not answer:
            continue
        rows.append({
            "focus": focus,
            "source": source,
            "qtype": qtype,
            "question": question,
            "answer": answer,
        })
    return rows


def main():
    all_rows = []
    for folder in sorted(os.listdir(REPO_DIR)):
        folder_path = os.path.join(REPO_DIR, folder)
        if not os.path.isdir(folder_path) or folder.startswith("."):
            continue
        folder_rows = []
        for fname in sorted(os.listdir(folder_path)):
            if not fname.endswith(".xml"):
                continue
            rows = parse_file(os.path.join(folder_path, fname))
            folder_rows.extend(rows)
            if len(folder_rows) >= MAX_PER_FOLDER:
                break
        print(f"{folder}: {len(folder_rows)} QA pairs")
        all_rows.extend(folder_rows)
        if len(all_rows) >= MAX_TOTAL:
            break

    all_rows = all_rows[:MAX_TOTAL]
    print(f"Total QA pairs: {len(all_rows)}")

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["focus", "source", "qtype", "question", "answer"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
