import pytest

from deep_mca.predict import predict
from deep_mca.utils import disassemble_hex

VOCAB_PATH = "data/vocab.pkl"

TEST_BLOCKS = [
    ("4889de4889c24c89ff", 91.0),
    ("418b4424084d8b3424498d2cc64939ee", 100.0),
    ("488b7d00be40000000", 56.0),
    ("4881f9308c8e00", 35.0),
]

ACCEPTABLE_ERROR = 0.3


def _hex_to_asm(hex_str: str) -> str:
    return "\n".join(disassemble_hex(hex_str))


def test_predict_returns_positive_float():
    asm = _hex_to_asm("4889de4889c24c89ff")
    result = predict(asm, vocab_path=VOCAB_PATH)
    assert isinstance(result, float)
    assert result > 0


@pytest.mark.parametrize("hex_str,ground_truth", TEST_BLOCKS)
def test_predict_TEST_BLOCKS(hex_str: str, ground_truth: float):
    asm = _hex_to_asm(hex_str)
    pred = predict(asm, vocab_path=VOCAB_PATH)
    relative_error = abs(pred - ground_truth) / ground_truth
    assert relative_error < ACCEPTABLE_ERROR, (
        f"Predicted {pred:.2f}, expected: {ground_truth:.2f} (relative error {relative_error:.2%})"
    )
