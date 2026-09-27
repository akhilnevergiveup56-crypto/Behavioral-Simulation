from src.data.jsonl_loader import load_jsonl


def test_load_jsonl():

    examples = load_jsonl("data/raw/behavioral_data.jsonl")

    assert len(examples) == 40

    example = examples[0]

    assert "emotions" in example
    assert "behavioral" in example
    assert "cognitive" in example
    assert "context" in example
    assert "scenario" in example
    assert "behavioral_targets" in example
    assert "context_story" in example