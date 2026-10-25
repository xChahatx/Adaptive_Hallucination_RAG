from src.verifier import verify_answer

chunks = [{"page": 9, "text": "Implementation Rules: 1. The examinations, attendance criteria to appear in examinations, promotion and award of the degree shall be governed by the Ordinance 11 of the University."}]

tests = {
    "correct": "Attendance criteria to appear in examinations are governed by Ordinance 11 of the University.",
    "wrong number": "Attendance criteria to appear in examinations are governed by Ordinance 15 of the University.",
    "absent": "The minimum attendance required to appear in examinations is 75% of lectures.",
}

for name, ans in tests.items():
    v = verify_answer(ans, chunks)
    for c in v["claims"]:
        print(f"{name:13} supported={c['supported']} score={c['best_score']} missing={c['missing']} entail={c['entail']}")