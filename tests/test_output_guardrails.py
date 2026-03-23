from models.rag_service import policy_check_output, grounding_check_output, append_disclaimer


def test_dosage_advice_blocked():
    text = "You should take 500 mg metformin daily."
    ok, reason = policy_check_output(text)
    assert ok is False
    assert "Dosage" in reason or "prescribing" in reason


def test_missing_disclaimer_fixed():
    text = "Trial NCT12345678 appears eligible."
    fixed = append_disclaimer(text)
    assert "consult your doctor" in fixed.lower() or "healthcare provider" in fixed.lower()


def test_hallucinated_trial_blocked():
    recommendation = "Trial NCT99999999 appears eligible."
    retrieved_trials = [{"nct_number": "NCT12345678"}]
    ok, reason = grounding_check_output(recommendation, retrieved_trials)
    assert ok is False