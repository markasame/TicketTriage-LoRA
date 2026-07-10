from tickettriage.eval.judge import judge_batch, judge_reply, summarize_scores
from tickettriage.schema import JudgeScore


class FakeParsed:
    def __init__(self, parsed_output):
        self.parsed_output = parsed_output


class FakeMessages:
    def __init__(self, score=None, fail=False):
        self.score = score or JudgeScore(relevance=4, tone=5, correctness=3, rationale="ok")
        self.fail = fail
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("api down")
        return FakeParsed(self.score)


class FakeClient:
    def __init__(self, **kw):
        self.messages = FakeMessages(**kw)


def test_judge_reply_passes_rubric_and_schema():
    client = FakeClient()
    score = judge_reply(client, "where is my order?", "Here is help...")
    assert score.mean == (4 + 5 + 3) / 3
    call = client.messages.calls[0]
    assert call["output_format"] is JudgeScore
    assert "RELEVANCE" in call["system"][0]["text"]
    assert "where is my order?" in call["messages"][0]["content"]


def test_judge_batch_survives_failures():
    client = FakeClient(fail=True)
    rows = [{"ticket": "t1", "reply": "r1"}, {"ticket": "t2", "reply": "r2"}]
    scores = judge_batch(client, rows, "reply")
    assert scores == [None, None]


def test_summarize_scores():
    scores = [
        JudgeScore(relevance=4, tone=4, correctness=4, rationale="a"),
        JudgeScore(relevance=2, tone=4, correctness=3, rationale="b"),
        None,
    ]
    summary = summarize_scores(scores)
    assert summary["n"] == 2
    assert summary["n_failed"] == 1
    assert summary["relevance"] == 3.0
    assert summary["tone"] == 4.0


def test_summarize_empty():
    assert summarize_scores([None])["n"] == 0
