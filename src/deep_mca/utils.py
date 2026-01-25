import subprocess


def disassemble(hex_str: str, output_intel_syntax: bool = False) -> str:
    """
    This function is adapted from disasm from bhive.
    """
    args = []
    for i in range(0, len(hex_str), 2):
        byte = hex_str[i : i + 2]
        args.append("0x" + byte)

    syntax_id = 1 if output_intel_syntax else 0
    cmd = "echo {} | llvm-mc -disassemble -triple=x86_64 -output-asm-variant={}".format(
        " ".join(args),
        syntax_id,
    )
    stdout = subprocess.check_output(cmd, shell=True)
    return stdout.decode("utf8")


def disassemble_hex(hex_str: str, output_intel_syntax: bool = False) -> list[str]:
    output = disassemble(hex_str, output_intel_syntax=output_intel_syntax)
    lines = []
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("."):
            continue
        lines.append(line)
    return lines


def wrap_asm(lines: list[str]) -> str:
    """
    Wrap basic block in a label so llvm-mca can parse it.
    """
    body = "\n  ".join(lines)
    return f""".text
.globl bb
bb:
  {body}
"""
