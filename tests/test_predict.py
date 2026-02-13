import pytest

from deep_mca.predict import predict

VOCAB_PATH = "data/vocab.pkl"

TEST_BLOCKS = [
    ("movq\t%rbx, %rsi\nmovq\t%rax, %rdx\nmovq\t%r15, %rdi", 91.0),
    ("movl\t8(%r12), %eax\nmovq\t(%r12), %r14\nleaq\t(%r14,%rax,8), %rbp\ncmpq\t%rbp, %r14", 100.0),
    ("movq\t(%rbp), %rdi\nmovl\t$64, %esi", 56.0),
    ("cmpq\t$9342000, %rcx", 35.0),
]

ACCEPTABLE_ERROR = 0.3


def test_predict_returns_positive_float():
    asm = "movq\t%rbx, %rsi\nmovq\t%rax, %rdx\nmovq\t%r15, %rdi"
    result = predict(asm, vocab_path=VOCAB_PATH)
    assert isinstance(result, float)
    assert result > 0


@pytest.mark.parametrize("asm,ground_truth", TEST_BLOCKS)
def test_predict_TEST_BLOCKS(asm: str, ground_truth: float):
    pred = predict(asm, vocab_path=VOCAB_PATH)
    relative_error = abs(pred - ground_truth) / ground_truth
    assert relative_error < ACCEPTABLE_ERROR, (
        f"Predicted {pred:.2f}, expected: {ground_truth:.2f} (relative error {relative_error:.2%})"
    )
