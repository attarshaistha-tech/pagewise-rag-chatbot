"""
eval.py - measure how well your chatbot answers, so you have a real number for your resume.

1. Build the index first (run the app and click 'Build index', or pass PDFs below).
2. Fill questions.csv with 15-20 questions and a keyword that a correct answer must contain.
3. Run: python eval.py
"""
import csv
import time

import rag

with open("questions.csv", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

correct = 0
for row in rows:
    answer, _ = rag.ask(row["question"])
    ok = row["expected_keyword"].lower() in answer.lower()
    correct += ok
    print("PASS" if ok else "FAIL", "-", row["question"])
    time.sleep(4)  # stay inside the free API rate limit

print(f"\nScore: {correct}/{len(rows)} = {100 * correct / len(rows):.0f}%")
