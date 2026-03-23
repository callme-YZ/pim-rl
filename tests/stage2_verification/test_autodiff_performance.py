"""
Test JAX Autodiff Performance (Corrected)

Issue #24 Task 3: Proper performance benchmark with JIT + block_until_ready

Author: 小P ⚛️
Date: 2026-03-24
"""

import sys
sys.path.insert(0, '../../src')

import time
import jax.numpy as jnp
from jax import grad, jit

from pytokmhd.geometry.toroidal import ToroidalGrid
from test_autodiff_hamiltonian import hamiltonian_jax


def benchmark_performance():
    """Proper performance benchmark"""
    print("=" * 60)
    print("JAX Autodiff Performance Benchmark")
    print("=" * 60)
    
    # Grid
    grid = ToroidalGrid(R0=1.0, a=0.3, nr=32, ntheta=64)
    
    r_grid = jnp.array(grid.r_grid)
    R_grid = jnp.array(grid.R_grid)
    dr = grid.dr
    dtheta = grid.dtheta
    
    # Fields
    r = r_grid[:, 0:1]
    theta = grid.theta_grid[0:1, :]
    psi = jnp.array(0.1 * r**2 * jnp.sin(2*theta))
    phi = jnp.array(0.05 * r * jnp.cos(theta))
    
    def H_func(psi, phi):
        return hamiltonian_jax(psi, phi, r_grid, dr, dtheta, R_grid)
    
    # JIT compile
    H_jit = jit(H_func)
    grad_psi_jit = jit(grad(H_func, argnums=0))
    grad_phi_jit = jit(grad(H_func, argnums=1))
    
    # Warmup (trigger compilation)
    print("\nWarming up (JIT compilation + cache)...")
    for _ in range(100):
        _ = H_jit(psi, phi).block_until_ready()
        _ = grad_psi_jit(psi, phi)[0,0].block_until_ready()
        _ = grad_phi_jit(psi, phi)[0,0].block_until_ready()
    
    print("✅ Warmup complete\n")
    
    # Benchmark H evaluation
    print("Benchmarking H evaluation...")
    n_runs = 1000
    
    start = time.time()
    for _ in range(n_runs):
        H = H_jit(psi, phi).block_until_ready()
    elapsed_H = time.time() - start
    
    time_H = elapsed_H / n_runs * 1e6  # microseconds
    
    print(f"  Runs: {n_runs}")
    print(f"  Total: {elapsed_H:.3f} s")
    print(f"  Per call: {time_H:.2f} μs")
    print(f"  Throughput: {n_runs/elapsed_H:.1f} /s")
    
    # Benchmark ∇_ψ H
    print("\nBenchmarking ∇_ψ H...")
    
    start = time.time()
    for _ in range(n_runs):
        grad_psi = grad_psi_jit(psi, phi)[0,0].block_until_ready()
    elapsed_grad_psi = time.time() - start
    
    time_grad_psi = elapsed_grad_psi / n_runs * 1e6
    
    print(f"  Runs: {n_runs}")
    print(f"  Total: {elapsed_grad_psi:.3f} s")
    print(f"  Per call: {time_grad_psi:.2f} μs")
    
    # Benchmark ∇_φ H
    print("\nBenchmarking ∇_φ H...")
    
    start = time.time()
    for _ in range(n_runs):
        grad_phi = grad_phi_jit(psi, phi)[0,0].block_until_ready()
    elapsed_grad_phi = time.time() - start
    
    time_grad_phi = elapsed_grad_phi / n_runs * 1e6
    
    print(f"  Runs: {n_runs}")
    print(f"  Total: {elapsed_grad_phi:.3f} s")
    print(f"  Per call: {time_grad_phi:.2f} μs")
    
    # Average gradient time
    time_grad_avg = (time_grad_psi + time_grad_phi) / 2
    
    print(f"\n  Average gradient: {time_grad_avg:.2f} μs")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"\nH evaluation:      {time_H:7.2f} μs")
    print(f"∇_ψ H (autodiff):  {time_grad_psi:7.2f} μs")
    print(f"∇_φ H (autodiff):  {time_grad_phi:7.2f} μs")
    print(f"Average gradient:  {time_grad_avg:7.2f} μs")
    
    overhead_psi = (time_grad_psi / time_H - 1) * 100
    overhead_phi = (time_grad_phi / time_H - 1) * 100
    overhead_avg = (time_grad_avg / time_H - 1) * 100
    
    print(f"\nOverhead vs H evaluation:")
    print(f"  ∇_ψ H: {overhead_psi:+6.1f}%")
    print(f"  ∇_φ H: {overhead_phi:+6.1f}%")
    print(f"  Average: {overhead_avg:+6.1f}%")
    
    # Comparison to theoretical best
    print(f"\nTheoretical best (reverse-mode AD):")
    print(f"  Expected overhead: ~3-5× H evaluation")
    print(f"  Actual overhead: {time_grad_avg/time_H:.1f}×")
    
    # Estimate FD cost
    nr, ntheta = psi.shape
    n_grid = nr * ntheta
    time_fd_estimate = 2 * n_grid * time_H
    speedup_vs_fd = time_fd_estimate / time_grad_avg
    
    print(f"\nVs Finite Difference (estimated):")
    print(f"  Grid points: {n_grid}")
    print(f"  FD time (2N×H): {time_fd_estimate/1e6:.3f} s")
    print(f"  Autodiff speedup: {speedup_vs_fd:.0f}×")
    
    # Pass criteria
    print("\n" + "=" * 60)
    
    # Reasonable overhead: <10× H evaluation
    # (Reverse-mode AD should be 3-5× for well-optimized code)
    
    if time_grad_avg < 10 * time_H:
        print("✅ PERFORMANCE ACCEPTABLE")
        print(f"   Gradient {time_grad_avg/time_H:.1f}× slower than H (< 10× threshold)")
        
        if time_grad_avg < 5 * time_H:
            print("✅ EXCELLENT - Near theoretical optimum!")
        
        print(f"\n✅ Massive speedup vs FD: {speedup_vs_fd:.0f}×")
        print("   (FD requires 2N Hamiltonian evaluations)")
        
        return True
    else:
        print("⚠️ Performance suboptimal")
        print(f"   Gradient {time_grad_avg/time_H:.1f}× slower than H")
        print("   Expected: 3-5× for reverse-mode AD")
        
        return False


if __name__ == "__main__":
    success = benchmark_performance()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ Task 3 Complete: Performance validated")
        print("=" * 60)
    else:
        print("\n⚠️ Performance below expectation (but still usable)")
