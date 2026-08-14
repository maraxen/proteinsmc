"""Conformance: this repo's alphabet constants match the ecosystem's declarations.

WHY THIS EXISTS

A silent data-corruption bug shipped from this file's constants: the ESM token table was built
from the AlphaFold residue ordering while carrying a ProteinMPNN name, so every caller honouring
the documented contract had its residues permuted (only A, S and T are fixed points of that
permutation). It was shape-valid, so nothing raised.

An ecosystem census found the same two base orderings declared under five different names across
four repos. The library `alphex` holds one declaration of each. This test asserts that
*this repo's* constants still agree with those declarations, so that if either side drifts, a
test fails instead of a number quietly changing.

This is Phase 0 of the migration (decision D4): a **dev-dependency-only** conformance check.
Nothing under `src/` imports the library, so no runtime dependency edge is created and the
ecosystem partition is unaffected.

If a test here fails, do NOT edit it to match. One of the two sides is wrong, and which one is
wrong is the finding.
"""

from __future__ import annotations

import pytest

from proteinsmc.utils import constants as C  # noqa: N812

alphex = pytest.importorskip(
  "alphex",
  reason="alphabet conformance library not installed; add it to the dev group",
)
known = alphex.known
SpecialKind = alphex.SpecialKind


def test_restypes_is_the_alphafold_ordering() -> None:
  """`restypes` is AlphaFold-ordered, despite living next to ProteinMPNN-named constants."""
  assert "".join(C.restypes) == known.AF_20.symbols


def test_proteinmpnn_restypes_is_the_proteinmpnn_ordering() -> None:
  assert C.PROTEINMPNN_RESTYPES == known.MPNN_20.symbols


def test_the_two_orderings_in_this_module_are_actually_different() -> None:
  """The premise of the whole bug: these look interchangeable and are not.

  Only A, S and T survive the permutation between them.
  """
  assert C.PROTEINMPNN_RESTYPES != "".join(C.restypes)
  fixed = [a for a, b in zip(C.PROTEINMPNN_RESTYPES, C.restypes, strict=True) if a == b]
  assert fixed == ["A", "S", "T"]


def test_aa_char_to_int_map_indexes_the_alphafold_ordering() -> None:
  """`AA_CHAR_TO_INT_MAP` is built from `restypes`, so its index space is AlphaFold's.

  This is the fact that architecture review finding #1 got wrong: the sentinel constant is
  named `PROTEINMPNN_X_INT`, but the space it inhabits is AlphaFold-ordered.
  """
  for symbol, index in C.AA_CHAR_TO_INT_MAP.items():
    assert known.AF_20.index_of(symbol) == index


def test_sentinel_indices_match_the_declared_22_wide_spaces() -> None:
  """Both 22-wide declarations put gap at 20 and stop/unknown at 21."""
  assert C.PROTEINMPNN_X_INT == 21
  for alphabet in (known.MPNN_GAP_X_STOP_22, known.AF_GAP_X_STOP_22):
    assert alphabet.specials[SpecialKind.UNKNOWN] == C.UNKNOWN_AA_INT
    assert alphabet.specials[SpecialKind.STOP] == C.STOP_INT
    assert alphabet.size == C.PROTEINMPNN_X_INT + 1


def test_stop_and_unknown_are_conflated_here_and_the_declaration_says_so() -> None:
  """A known defect, declared rather than hidden.

  `STOP_INT == UNKNOWN_AA_INT == 21` means a stop codon and an unknown residue are
  indistinguishable at runtime. The declaration reports it via `conflated_specials` and lints
  dirty, which is strictly more visibility than three consecutive assignments provide.
  """
  assert C.STOP_INT == C.UNKNOWN_AA_INT
  declaration = known.AF_GAP_X_STOP_22
  assert declaration.conflated_specials == frozenset(
    {frozenset({SpecialKind.UNKNOWN, SpecialKind.STOP})},
  )
  assert declaration.lint()


def test_esm_vocabulary_ordering_matches_the_declaration() -> None:
  """ESM's 20 canonical residues are contiguous at offset 4, in a third distinct ordering."""
  residues = [c for c in C.ESM_SEQUENCE_VOCAB if len(c) == 1 and c.isalpha() and c.isupper()]
  esm_20 = "".join(residues[: known.ESM_C.n_symbols])
  assert esm_20 == known.ESM_C.symbols
  assert C.ESM_SEQUENCE_VOCAB.index("L") == known.ESM_C.offset
  assert C.ESM_AA_CHAR_TO_INT_MAP["-"] == known.ESM_C.specials[SpecialKind.GAP]
  assert C.ESM_MASK_ID == known.ESM_C.specials[SpecialKind.MASK]
  assert C.ESM_BOS_ID == known.ESM_C.specials[SpecialKind.BOS]
  assert C.ESM_EOS_ID == known.ESM_C.specials[SpecialKind.EOS]


def test_nucleotide_alphabet_matches_the_declaration() -> None:
  assert "".join(C.NUCLEOTIDES_CHAR) == known.DNA_4.symbols


def test_the_two_esm_token_maps_disagree_exactly_where_the_orderings_do() -> None:
  """The fix that closed the original bug, pinned.

  Two maps now exist, one per source ordering. They must differ at every index where the two
  orderings differ, and agree nowhere else -- if they ever became equal, the disambiguation
  would have been silently undone.
  """
  mpnn_map = C.PROTEINMPNN_TO_ESM_AA_MAP_JAX
  af_map = C.ALPHAFOLD_TO_ESM_AA_MAP_JAX
  differ = [i for i in range(known.MPNN_20.n_symbols) if int(mpnn_map[i]) != int(af_map[i])]
  expected = [
    i
    for i in range(known.MPNN_20.n_symbols)
    if known.MPNN_20.symbols[i] != known.AF_20.symbols[i]
  ]
  assert differ == expected
  assert len(differ) == 17
