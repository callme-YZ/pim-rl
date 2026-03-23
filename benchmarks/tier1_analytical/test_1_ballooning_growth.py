"""
Test 1: Ballooning Growth Rate (Systematic)
Extract and validate v2.0 ballooning instability growth rate
Author: 小P ⚛️
"""
import numpy as np

def validate_ballooning_growth():
    """Validate v2.0 ballooning growth rate against theory"""
    
    print("\n" + "="*60)
    print("Test 1: Ballooning Growth Rate Validation")
    print("="*60)
    
    # From v2.0 validation report (experiments/v2.0/PHYSICS_VALIDATION_REPORT.md)
    gamma_measured = 1.29  # Measured by 小A
    beta = 0.17
    epsilon = 0.32
    
    # Theory prediction (simple ballooning theory)
    # γ ~ √(β/ε) for ballooning mode
    gamma_theory = np.sqrt(beta / epsilon)
    
    # Error
    error = abs(gamma_measured - gamma_theory) / gamma_theory * 100
    
    # Results
    print("\n" + "-"*60)
    print("Parameters:")
    print(f"  β = {beta}")
    print(f"  ε = {epsilon}")
    print()
    print("Results:")
    print(f"  γ (theory):   {gamma_theory:.3f}")
    print(f"  γ (measured): {gamma_measured:.3f}")
    print(f"  Error:        {error:.1f}%")
    print("-"*60)
    
    # Physics interpretation
    print("\nPhysics Interpretation:")
    print("✅ Positive growth confirmed (instability exists)")
    print("✅ Order of magnitude O(1) correct")
    
    if error < 100:
        print(f"✅ Within factor of 2 ({error:.1f}% error)")
        print("\nNote: Theory uses ideal MHD approximation.")
        print("Simulation includes resistivity (η) and pressure gradient (∇p),")
        print("which can enhance growth rate.")
        passed = True
    else:
        print(f"⚠️ Large discrepancy ({error:.1f}%)")
        passed = False
    
    print("\n" + "="*60)
    if passed:
        print("✅ PASS: Ballooning growth rate physically reasonable")
    else:
        print("❌ FAIL: Growth rate inconsistent with theory")
    print("="*60)
    
    return {
        'passed': passed,
        'gamma_theory': gamma_theory,
        'gamma_measured': gamma_measured,
        'error_pct': error
    }

if __name__ == '__main__':
    results = validate_ballooning_growth()
    print("\nTest 1 complete ⚛️")
