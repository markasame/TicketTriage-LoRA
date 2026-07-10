from tickettriage.eval.capability import (
    load_questions,
    parse_answer,
    question_prompt,
    run_capability_check,
)


def test_questions_wellformed():
    questions = load_questions()
    assert len(questions) >= 20
    for item in questions:
        assert len(item["choices"]) == 4
        assert item["answer"] in "ABCD"


def test_question_prompt_contains_choices():
    item = load_questions()[0]
    prompt = question_prompt(item)
    for choice in item["choices"]:
        assert choice in prompt


def test_parse_answer():
    assert parse_answer("B") == "B"
    assert parse_answer("The answer is C.") == "C"
    assert parse_answer("  d\n") == "D"
    assert parse_answer("no letter here") is None


def test_run_capability_check_with_fixed_backend():
    class AlwaysB:
        def generate(self, prompt, max_new_tokens=10):
            return "B"

    result = run_capability_check(AlwaysB())
    questions = load_questions()
    expected = sum(1 for q in questions if q["answer"] == "B") / len(questions)
    assert result["accuracy"] == round(expected, 4)
    assert result["n"] == len(questions)
