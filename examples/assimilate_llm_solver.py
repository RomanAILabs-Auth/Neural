# Works both ways:
#   nrlpy run examples/assimilate_llm_solver.py
#   python examples/assimilate_llm_solver.py   (needs PYTHONPATH=nrlpy/src)

try:
    nrl  # type: ignore[name-defined]
except NameError:
    from nrlpy.compat import nrl

neurons = 4096
pot = bytearray(nrl.packed_bytes(neurons))
inp = bytearray(nrl.packed_bytes(neurons))
for i in range(len(inp)):
    inp[i] = ((i * 17) & 0x77) & 0xFF

out = nrl.assimilate(pot, inp, neurons=neurons, iterations=256, threshold=10)
cli = nrl.assimilate_cli(neurons, 256, 10)
assert out["checksum_fnv1a64"] == cli["checksum_fnv1a64"], "Python buffers == nrl assimilate"
print("assimilation OK", out["variant"], "checksum", out["checksum_fnv1a64"])
