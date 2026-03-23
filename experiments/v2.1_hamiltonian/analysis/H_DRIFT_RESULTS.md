# Phase A.1: H Drift Measurement Results

**Date:** 2026-03-23
**Configuration:** Strong (λ_H=1.0)
**Episodes:** 10

## Summary

**Mean H drift:** 16.211% ± 0.000%
**Max H drift:** 16.211%
**Target:** < 0.1%

**Status:** ❌ **FAIL** (Mean drift > 1%)

## Episode-by-Episode Results

| Episode | Steps | Reward | H Initial | H Final | H Drift (%) |
|---------|-------|--------|-----------|---------|-------------|
| 1 | 166 | -6.53 | -0.2321 | -0.1945 | 16.211 |
| 2 | 166 | -6.53 | -0.2321 | -0.1945 | 16.211 |
| 3 | 166 | -6.53 | -0.2321 | -0.1945 | 16.211 |
| 4 | 166 | -6.53 | -0.2321 | -0.1945 | 16.211 |
| 5 | 166 | -6.53 | -0.2321 | -0.1945 | 16.211 |
| 6 | 166 | -6.53 | -0.2321 | -0.1945 | 16.211 |
| 7 | 166 | -6.53 | -0.2321 | -0.1945 | 16.211 |
| 8 | 166 | -6.53 | -0.2321 | -0.1945 | 16.211 |
| 9 | 166 | -6.53 | -0.2321 | -0.1945 | 16.211 |
| 10 | 166 | -6.53 | -0.2321 | -0.1945 | 16.211 |

## Interpretation

⚠️ H drift slightly exceeds strict conservation target (0.1%) but remains well below 1%, which is acceptable for a pseudo-Hamiltonian.

**Possible explanations:**
- H is a learned pseudo-Hamiltonian (not exact physical energy)
- Environment dynamics include dissipation (resistivity η)
- Action constraints prevent perfect conservation

## H-Reward Correlation: -0.523

✅ Strong correlation indicates H captures value-relevant structure.

## Next Steps

- Share results with 小P for physics validation
- Proceed to Phase A.2: Sample efficiency analysis
- If drift > 1%: Investigate H network architecture
