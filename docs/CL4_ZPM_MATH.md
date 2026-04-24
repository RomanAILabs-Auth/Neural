<!-- Copyright (c) 2026 Daniel Harding - RomanAILabs. All Rights Reserved. -->

# Cl(4,0) ↔ ZPM: the math as it exists in `zpm.py` today

> **Status.** Descriptive. This document explains what the code actually
> does, not what a more ambitious version could do. Every claim here is
> anchored to a line range in `nrlpy/src/nrlpy/zpm.py` or
> `nrlpy/src/nrlpy/lmo.py`. If a term is not in those files, it is
> defined here and only here.

## 1. The object: a 256-bit `State`

The ZPM's `State` is four 64-bit unsigned words — a single
`tuple[int, int, int, int]`:

```71:72:nrlpy/src/nrlpy/zpm.py
# 4 × uint64 = 256-bit topological state (public type alias).
State = tuple[int, int, int, int]
```

Call these words $w_0, w_1, w_2, w_3 \in \{0, \ldots, 2^{64}-1\}$. The
256-bit manifold that code comments refer to is literally this
concatenation, treated as $\mathrm{GF}(2)^{256}$ when we do XOR/Hamming
work and as four independent Galois projections when we do anchoring.

There is **no** floating-point $\mathbb{R}^4$ space anywhere in the cache
path. The Cl(4,0) bivectors live in a different layer — see §4.

## 2. Stage I — `anchor`: four rotated FNV-1a64 projections

`anchor(blob)` fingerprints a turn bundle
(`model_sha256 | system | history | user | sampler`) into a `State`:

```104:132:nrlpy/src/nrlpy/zpm.py
def anchor(turn_bytes: bytes | Iterable[bytes]) -> State:
    """Materialize the 4 × uint64 manifold anchor for a turn.
    ...
    """
    ...
    w0 = _fnv1a64(blob)
    w1 = _fnv1a64(blob[::-1])
    half = n // 2
    w2 = _fnv1a64(blob[half:] + blob[:half])
    shift = n // 4 if n >= 4 else 1
    w3 = _fnv1a64(blob[shift:] + blob[:shift])
    return (w0, w1, w2, w3)
```

Each $w_i$ is FNV-1a64 over a deterministic permutation of the same
bytes (identity, reverse, halves-swap, quarter-shift). The crucial
property: a single-byte change in the blob flips ≈ 50% of the bits in
each of the four words **independently** (FNV's avalanche), so small
perturbations in the bundle produce a state roughly 128 bits apart in
Hamming space. Rephrases that share much of the bundle don't land close
together — which is exactly why R1/R2 primary lookups have been missing
on rephrases, and why the Phase 11 rescue reads from `prompt_head`
rather than trying to tighten the anchor.

## 3. Stage II — `inversion`: GF(2) row parity

```139:151:nrlpy/src/nrlpy/zpm.py
def inversion(t: State) -> State:
    """GF(2) row parity matching ``zpm.cpp``'s inversion():

    ``rows[i] = t[i] ^ (t[i] << 1) ^ 0x9E3779B97F4A7C15``
    ...
    """
    return tuple(
        ((t[i] ^ ((t[i] << 1) & _U64_MASK) ^ ZPM_GOLDEN_RATIO) & _U64_MASK)
        for i in range(4)
    )
```

Per word: $r_i = w_i \oplus (w_i \ll 1) \oplus \varphi_{64}$ where
$\varphi_{64} = \texttt{0x9E3779B97F4A7C15}$ is the 64-bit truncation of
the golden-ratio conjugate. Two observations:

1. **$w_i \oplus (w_i \ll 1)$ is the discrete derivative in $\mathrm{GF}(2)$.**
   Bit $b$ of the result is $w_i[b] \oplus w_i[b-1]$, i.e. the
   adjacent-bit difference. This is the finite-field analogue of $dw/dx$.

2. **The XOR with $\varphi_{64}$ breaks periodicity.** Without it,
   near-identity inputs ($w_i \approx 0$ or $w_i \approx \mathrm{all\ ones}$)
   collapse to trivial rows and every bundle with a constant prefix would
   alias. The golden-ratio constant is irrational in $\mathbb{Z}/2\mathbb{Z}$
   in the sense that its base-2 expansion has no short period, so row-level
   distinguishability is preserved.

`inversion_determinant` is then $r_0 \oplus r_1 \oplus r_2 \oplus r_3$
(line 156); non-zero means the rows do not collectively cancel, i.e. the
state is a "unity signature" rather than a trivial fixed point of the
inversion map.

## 4. Stage III — `rotor`: a Cl(4,0) sandwich, but only as a scalar seed

Here is where "Clifford Algebra" enters, and it is narrower than the
reference docs imply. The `Rotor` is a 4-component float struct:

```164:180:nrlpy/src/nrlpy/zpm.py
@dataclass(frozen=True)
class Rotor:
    """Clifford Cl(4,0) rotor components. ``s`` scalar, ``b_*`` bivectors."""

    s: float
    b_xy: float
    b_xz: float = 0.7071067811865475  # Fixed spacetime anchor from ROMA-ZPM v2.0
    b_xw: float = 0.5  # Hyper-dimensional offset

    @property
    def norm(self) -> float:
        return (self.s * self.s + self.b_xy * self.b_xy) ** 0.5
```

In Clifford algebra Cl(4,0) — the geometric algebra on $\mathbb{R}^4$
with positive-definite quadratic form — a rotor is an even-grade
element $R = s + b_{xy}\,e_{xy} + b_{xz}\,e_{xz} + b_{xw}\,e_{xw}
+ \ldots$ and it acts on a vector $v$ via the **sandwich product**
$v' = R\,v\,\tilde R$. A *proper* rotor also has $R\tilde R = 1$,
i.e. unit norm.

What the code does: it populates $s$ and $b_{xy}$ from a 64-bit seed,
fixes $b_{xz}$ and $b_{xw}$ to constants, and only ever checks
`self.norm > 0.999` (line 178–179). The sandwich product
$R\,v\,\tilde R$ is **never evaluated on `State` words**. There is no
`v = w_0 e_0 + w_1 e_1 + w_2 e_2 + w_3 e_3` vector being rotated by
`R`. The rotor is a *phase-lock indicator*: a seed whose
$(s = \cos x, b_{xy} = \sin x)$ pair is close to unit norm is considered
"phase-locked". The fixed $b_{xz}, b_{xw}$ are labelled "spacetime
anchor" / "hyper-dimensional offset" and carry no dynamic meaning in the
current codebase — they are type tags.

```182:192:nrlpy/src/nrlpy/zpm.py
def rotor(seed: int) -> Rotor:
    """Build the rotor from the XOR of the first two inversion rows.
    ...
    """
    import math

    x = (seed & _U64_MASK) * 1e-9
    return Rotor(s=math.cos(x), b_xy=math.sin(x))
```

So the honest statement is: **Cl(4,0) enters the code as a typed
identity for a scalar phase check, not as an active rotation operator
on the 256-bit state.** The 256-bit state is never rotated by the
rotor; it is transformed by `inversion()` (GF(2)) and compared by
`hamming_state()` / XOR.

This matters for the rescue discussion below: when I wrote
"Stage-VI skips bit-symmetry" I meant the XOR residual check. There is
no geometric/floating rotation to also bypass — there wasn't one in the
serving path to begin with.

## 5. Stage IV — `nullspace_search`: XOR + popcount, nothing more

```216:245:nrlpy/src/nrlpy/zpm.py
def nullspace_search(
    query: State,
    stored: Sequence[State],
    *,
    threshold_bits: int = 0,
) -> ZpmHit:
    """Scan stored unity states; return nearest + Hamming distance.
    ...
    """
    ...
    best_d = hamming_state(query, stored[0])
    for i in range(1, len(stored)):
        d = hamming_state(query, stored[i])
        if d < best_d:
            best_i = i
            best_d = d
```

where

```84:86:nrlpy/src/nrlpy/zpm.py
def hamming_state(a: State, b: State) -> int:
    """Hamming distance between two 256-bit states (``Σ popcount(a[i] ^ b[i])``)."""
    return sum(popcount64(a[i] ^ b[i]) for i in range(4))
```

The metric is the $\mathrm{GF}(2)^{256}$ Hamming distance:

$$d(a, b) = \sum_{i=0}^{3} \mathrm{popcount}(a_i \oplus b_i) \in [0, 256]$$

`exact` means $d = 0$ (perfect match of all 256 bits). `within_threshold`
means $d \le \texttt{threshold\_bits}$, tuned to 28 in rewired mode — so
we accept states that differ in ≤ 28 of 256 bits (≈11%).

## 6. Stage VI — `verify`: the OR of word-XOR residuals

```253:264:nrlpy/src/nrlpy/zpm.py
def verify(t: State, solution: State) -> bool:
    """Bit-symmetry audit. Residual is the OR of per-word XOR diffs.
    ...
    """
    residual = 0
    for i in range(4):
        residual |= (t[i] ^ solution[i]) & _U64_MASK
    return residual == 0
```

`verify` returns `True` iff $t = s$ exactly — it is strictly stricter
than `nullspace_search`'s `within_threshold`. Even `distance_bits=1`
fails `verify`. **This is the fuse.** Any cache served through the
standard path has been proved bit-identical to a stored state.

## 7. R2 Omega: how the chain composes

R2's pipeline in `try_omega_native_resolve` (see `lmo.py:2144+`):

1. `_run_omega_probe(lmo, intent_anchor_bytes)` projects the intent
   bytes onto every sub-lattice in the LMO router graph, does a bounded
   `inversion → rotor-phase-check` per sub-lattice, and folds the
   per-sub-lattice digests into a 256-bit `candidate_state` via a
   Plane-A.5 combine (four FNV-1a64 passes over the concatenated
   digests — same shape as `anchor()`, §2).

2. `zpm_index.lookup(candidate_state, threshold_bits)` is the
   `nullspace_search` of §5.

3. On a primary hit, `_stage_vi_shadow_audit` runs `zpm.verify` among
   other checks. If verify fails, R2 demotes to R5.

The critical observation for Phase 11 is a **write/read state
mismatch**:

- **R5 writeback** (the only path that adds entries to the live
  ZPM index in a real chat) stores entries at
  `state = zpm.anchor(_zpm_anchor_bytes(manifest, prompt))` — the
  R1 anchor, §2. `gguf.py:1595+`.
- **R2 primary lookup** queries at the R2 Omega-folded state from
  step 1 above. `lmo.py:2144+`.

These two states are computed by different pipelines and have no
reason to agree. Hamming distance between them is empirically ≈ 100
bits on the bench corpus (the audit shows `distance=117..140b`), far
above any reasonable `zpm_threshold_bits`. **R2 primary therefore
cannot hit any R5-written entry.** In production it only ever hit
entries primed by external tooling (e.g. final_wps_bench). That
explains the 5% R2 share the benchmarks previously reported.

## 8. Why the n-gram rescue bridges this gap

The Phase 11 rescue path (`_ngram_rescue_search` in `lmo.py:2050+`)
sidesteps the state mismatch entirely:

1. It iterates over stored ZPM entries (up to `_NGRAM_MAX_SCAN = 256`).
2. For each entry it reads `prompt_head` from metadata (the first
   256 bytes of the prompt that produced the entry, stamped by the
   R5 writeback at `gguf.py:1596+`).
3. It computes char-3-gram Jaccard overlap between the current
   `prompt_text` and `prompt_head`.
4. The best entry above `_NGRAM_OVERLAP_THRESHOLD = 0.30` is admitted.

The rescue admission audit (`_stage_vi_ngram_rescue_audit`)
**deliberately omits `zpm.verify`**, because by construction the
rescued entry's stored state ≠ the R2 candidate state — verify would
always reject. Instead, admission is gated on:

- `prompt_head` overlap ≥ 0.30 (surface similarity),
- tokenizer/UTF-8/blob integrity checks (the non-verify parts of
  Stage-VI still run).

**What this buys.** Rescued turns get served when the stored prompt
was a plausible rephrasing of the current prompt — measured by
character surface overlap, not by the 256-bit state.

**What it costs honestly.**

- No bit-symmetry guarantee. A rescue is *not* a proof the reply is
  correct for this prompt; it's a statement that the stored prompt
  was surface-similar. If the 0.30 threshold admits a spurious match,
  the reply text will still be the stored one.
- The rescue is flagged in every evidence log with `ngram_rescued=True`
  and surfaced in the chat footer as `[Omega Rescue · overlap=…]`, so
  operators can spot-check rescue quality during live sessions.
- `r2_rescue_bench.py` gives the bulk-audit view: on 63 curated
  rephrasings the rescue fires on 47 (74.6% served share) with 0.31
  min / 0.43 p50 / 0.75 max overlap. Zero of the 47 produced
  cross-topic matches.

## 9. Where Cl(4,0) could matter, and doesn't yet

If someone wants to revisit the rotor's role: the `b_xz = 2^{-1/2}`,
`b_xw = 0.5` constants, and the `phase_locked` norm check, are all
scaffolding for a sandwich product $R \cdot v \cdot \tilde R$ that
would rotate the four-word `State` in $\mathbb{R}^4$. Nothing in the
current cache path performs that rotation. If it were added, a
natural place would be between Stage II (inversion) and Stage IV
(nullspace search) — rotating the state into a canonical orientation
so that small perturbations of the source bundle produce states that
are *geometrically* close rather than just Hamming-close. That is a
real research direction, but it is not a description of today's code.

For the current system, the honest statement is: **"Cl(4,0)" labels
the rotor struct and the golden-ratio GF(2) step pattern in
`inversion`; the cache fuse is XOR/popcount Hamming on four 64-bit
words, and the rescue path works because `prompt_head` metadata is a
second, surface-level index onto the same entries.**
