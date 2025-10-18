from src.persona import PERSONA_QUESTIONS, build_persona_system_prompt


def test_build_persona_system_prompt_includes_only_filled_answers():
    answers = {
        "tone": "Warm and encouraging",
        "formality": "Casual",
        "pacing": "Measured",
        "vocabulary": "Accessible",
        "temperament": "Calm",
        "filler": "Avoid fillers",
        "directness": "Direct",
        "cultural_cues": "Reference hiking culture",
    }
    prompt = build_persona_system_prompt(answers)
    assert prompt.startswith("You are a realtime voice assistant")
    for question in PERSONA_QUESTIONS:
        label = question["label"]
        if answers.get(question["key"]):
            assert f"- {label}: {answers[question['key']]}" in prompt


def test_prompt_has_final_line():
    prompt = build_persona_system_prompt({})
    assert prompt.splitlines()[-1] == "Keep answers brief unless asked."
