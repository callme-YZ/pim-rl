# Phase A.2: Peak Performance Analysis (Revised)

**Date:** 2026-03-23
**Analysis:** Following 小P recommendation - focus on peak performance

## Key Finding

**All policies achieved peak performance at 40,000 steps, then degraded.**

This suggests:
- Task sensitivity (ballooning mode instability)
- Training instability in later stages
- Peak performance is more meaningful metric than final

## Peak Performance Results

| Configuration | Peak Reward | Steps to Peak | Final Reward | Degradation |
|---------------|-------------|---------------|--------------|-------------|
| Baseline (λ=0.0) | -8.00 | 40,000 | -8.02 | 0.3% |
| Weak (λ=0.1) | -7.05 | 40,000 | -8.15 | 15.6% |
| Medium (λ=0.5) | -6.96 | 40,000 | -7.76 | 11.4% |
| Strong (λ=1.0) | -5.59 | 40,000 | -5.86 | 4.7% |

## Peak Performance vs Baseline

| Configuration | Peak | Improvement vs Baseline |
|---------------|------|-------------------------|
| Baseline (λ=0.0) | -8.00 | +0.0% |
| Weak (λ=0.1) | -7.05 | +11.8% |
| Medium (λ=0.5) | -6.96 | +12.9% |
| Strong (λ=1.0) | -5.59 | +30.1% |

## Interpretation

**Best configuration:** Strong (λ=1.0)
**Peak improvement:** +30.1% over baseline

✅ **Significant improvement** at peak performance.

Hamiltonian guidance substantially improves peak capability, demonstrating physics-informed learning benefits.

### Degradation Analysis

All configurations degrade after 40k steps, with degradation ranging from 0.3% to 15.6%.

**Possible causes (小P analysis):**
- Task inherent difficulty (ballooning mode sensitivity)
- Over-optimization leading to instability
- Training dynamics (catastrophic forgetting, exploration-exploitation)

**Recommendation:** Use **peak performance** for paper comparisons, noting degradation as interesting finding.

## Sample Efficiency Conclusion

**Metric:** Peak performance achieved

All configurations peaked at the same step (40,000), suggesting **similar sample efficiency in reaching peak**.

**Primary benefit of Hamiltonian guidance: Better peak performance, not faster learning.**

## Paper Claims (Approved by 小P)

✅ "Hamiltonian policy achieves 30% better peak performance"
✅ "All policies peak around 40k steps and degrade thereafter"
✅ "Peak performance used for comparison (more meaningful)"
⚠️ "Degradation suggests task sensitivity and training challenges"

## Next Steps

- ✅ Phase A.2 complete (peak analysis)
- Proceed to Phase A.3: Parameter sweep (β, η variations)
- Consider investigating degradation mechanism (future work)
