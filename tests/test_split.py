from src.data.split import split_dataset


def test_split_dataset():

    examples = [{"id": i} for i in range(100)]

    train, val, test = split_dataset(examples)

    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15

    all_ids = {
        example["id"]
        for example in train + val + test
    }

    assert len(all_ids) == 100