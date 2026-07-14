"""
build_dataset.py
-----------------
Parses the raw MedQuAD XML files (organized in per-source folders) into a
single flat CSV file: data/medquad.csv

Columns: qid, question, answer, focus, source, qtype, url

Usage:
    python build_dataset.py --medquad_dir /path/to/MedQuAD --out data/medquad.csv
"""
import argparse
import csv
import glob
import os
import xml.etree.ElementTree as ET


def parse_file(path):
    """Parse a single MedQuAD XML document, yielding one dict per QA pair."""
    try:
        tree = ET.parse(path)
    except ET.ParseError:
        return  # a handful of files in MedQuAD have minor XML issues; skip them
    root = tree.getroot()

    focus = root.findtext("Focus", default="").strip()
    source = root.attrib.get("source", "")
    url = root.attrib.get("url", "")

    qa_pairs = root.find("QAPairs")
    if qa_pairs is None:
        return

    for qa in qa_pairs.findall("QAPair"):
        q_el = qa.find("Question")
        a_el = qa.find("Answer")
        if q_el is None or a_el is None:
            continue
        question = (q_el.text or "").strip()
        answer = (a_el.text or "").strip()
        if not question or not answer:
            continue
        yield {
            "qid": q_el.attrib.get("qid", ""),
            "question": question,
            "answer": answer,
            "focus": focus,
            "source": source,
            "qtype": q_el.attrib.get("qtype", ""),
            "url": url,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--medquad_dir", default="../MedQuAD",
                         help="Path to the cloned MedQuAD repo root")
    parser.add_argument("--out", default="data/medquad.csv",
                         help="Output CSV path")
    args = parser.parse_args()

    xml_files = glob.glob(os.path.join(args.medquad_dir, "**", "*.xml"), recursive=True)
    print(f"Found {len(xml_files)} XML files under {args.medquad_dir}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows_written = 0
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["qid", "question", "answer", "focus", "source", "qtype", "url"]
        )
        writer.writeheader()
        for path in xml_files:
            for row in parse_file(path):
                writer.writerow(row)
                rows_written += 1

    print(f"Wrote {rows_written} QA pairs to {args.out}")


if __name__ == "__main__":
    main()
