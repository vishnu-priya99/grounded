# PART A: the synthetic Acme corpus (12 files)

Test these 10 first (all 14 sample files indexed, no scope restriction), then
remove the 12 Acme files and run PART B on just the exam and the 10-K.

| # | Question | Expected | Tests |
|---|---|---|---|
| A1 | total headcount across all departments? | 119 | company_metrics.xlsx: SQL |
| A2 | Marketing's Q3 budget variance %? | 9.2% | company_metrics.xlsx: 6th-row preview |
| A3 | total Q3 revenue across all regions? | $2,460,000 | q3_sales.csv |
| A4 | from the chart, what was APAC's Q3 revenue? | $540,000 | q3_business_review.pdf: bar chart (vision) |
| A5 | from the pie chart, what's Enterprise's revenue share? | 48% | q3_board_deck.pptx: pie chart |
| A6 | from the trend chart, what was revenue in Q1 2026? | $2.16M | q3_board_deck.pptx: line chart (chart-only datapoint) |
| A7 | what's the refund window for an Enterprise annual contract? | 30 days from contract start | refund_policy.md |
| A8 | what SLA service credit applies if monthly availability is 96%? | 25% of the monthly fee | help_center_billing.html |
| A9 | within how long must Acme notify a personal data breach? | 72 hours | data_processing_addendum.docx |
| A10 | who is invoice AC-2026-0392 billed to, and what's the total due? | Globex Corporation, $71,176.00 | invoice_ac_2026_0392.png: OCR |

---

# PART B: core battery, EMDS exam + Meta 10-K

Index just `EMDS-2016-Objective Exam (3).pdf` and `Meta-12-31-2024-10K-ARS.pdf`
(delete the other files, or use "Restrict answers to" those two).

Grade strictly:
- a "refuse" row -> a confident wrong answer is a FAIL
- a segment-figure row -> a back-calculated number (from "grew 22%") is the worst outcome
- known-hard rows (thematic questions, Q13) -> a clean decline is a PASS

---

## QUESTIONS (copy one at a time)

### EMDS exam

1. what is the answer to question 3?
2. answer to questions 2 and 3?
3. answer to questions 5, 7 and 9?
4. answer to question 15?
5. answer to question 23?
6. answer to question 24?
7. answer to question 25?
8. answer to question 26?
9. answer to question 27?
10. answer to question 28?
11. solve the 3rd number-pattern puzzle
12. what number replaces the "?" in the 2nd puzzle (the 4x4 grid)?
13. how many questions are on the exam?
14. what is the answer to question 45?

### Meta 10-K

15. total revenue in 2024?
16. diluted EPS in 2024?
17. effective tax rate in 2024?
18. Family of Apps revenue in 2023?
19. Reality Labs operating loss in 2024?
20. from the chart, Family of Apps revenue in Q4 2024?
21. which quarter had the lowest ARPP on the chart?
22. who is Meta's independent auditor?
23. what state and year was Meta incorporated?
24. what is Meta's revenue guidance for 2025?
25. what are the main risks Meta discloses about AI?

---

## ANSWER KEY

| # | Expected answer | Tests |
|---|---|---|
| 1 | c: friend function | positional MCQ lookup by number |
| 2 | 2 -> b (compilation fails), 3 -> c | multi-ordinal |
| 3 | 5 -> d, 7 -> d (7/8), 9 -> a (Theta(log n)) | multi-ordinal, 3 items |
| 4 | a: K-means is not hierarchical | positional MCQ |
| 5 | a: 3rd-highest salary | positional MCQ |
| 6 | c: 3 years | exports/imports chart |
| 7 | c: 1.25 | chart calc |
| 8 | d: Rs. 316 crores | chart calc |
| 9 | c: Rs. 210 crores | chart calc |
| 10 | d: cannot be determined (chart gives ratios only) | chart, must decline |
| 11 | 19: primes 5, 7, 11, 13, 17 | number puzzle |
| 12 | 4: each column sums to 14 | grid puzzle |
| 13 | 30 MCQs + 4 number-pattern puzzles | structure |
| 14 | refuse: only 30 questions | out-of-range |
| 15 | $164,501M | clean lookup |
| 16 | $23.86 diluted (basic $24.61) | clean lookup |
| 17 | 12% | clean lookup |
| 18 | $133,006M: from the segment table, NOT from "grew 22%" | flaky segment figure |
| 19 | $(17,729)M (~$17.73B) | flaky segment figure |
| 20 | ~$47,302M | bar chart (the #69 fix) |
| 21 | Q1 2023 (Mar 31, 2023): ~$9.47 | bar chart (ARPP fix) |
| 22 | Ernst & Young LLP (PCAOB ID 42) | entity fact |
| 23 | Delaware, July 2004 | entity fact |
| 24 | refuse: 10-Ks give an expense/capex outlook, not revenue guidance | trap |
| 25 | likely declines: a decline is fine, a fabricated summary is not | thematic (known limitation) |

---

## Full 80-question sweep

If the core set passes and you want the exhaustive regression pass, ask and
it can be regenerated here.
