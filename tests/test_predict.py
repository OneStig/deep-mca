import pytest

from deep_mca.predict import predict

TEST_BLOCKS = [
    ("4889de4889c24c89ff", 91.0),
    ("418b4424084d8b3424498d2cc64939ee", 100.0),
    ("488b7d00be40000000", 56.0),
    ("4881f9308c8e00", 35.0),
]

ACCEPTABLE_ERROR = 0.3


def test_predict_returns_positive_float():
    result = predict("4889de4889c24c89ff")
    assert isinstance(result, float)
    assert result > 0


@pytest.mark.parametrize("hex_str,ground_truth", TEST_BLOCKS)
def test_predict_TEST_BLOCKS(hex_str: str, ground_truth: float):
    pred = predict(hex_str)
    relative_error = abs(pred - ground_truth) / ground_truth
    assert relative_error < ACCEPTABLE_ERROR, (
        f"Predicted {pred:.2f}, expected: {ground_truth:.2f} (relative error {relative_error:.2%})"
    )
