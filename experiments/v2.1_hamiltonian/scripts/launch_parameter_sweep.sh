#!/bin/bash
# Phase A.3: Launch Parameter Sweep
# 18 trainings: 9 (β,η) × 2 configs (baseline + Hamiltonian)

# Grid parameters (小P approved)
BETAS=(0.10 0.17 0.25)
ETAS=(0.005 0.01 0.02)
LAMBDAS=(0.0 1.0)

STEPS=50000

echo "========================================="
echo "Phase A.3: Parameter Sweep Launch"
echo "========================================="
echo "Grid: β ∈ {0.10, 0.17, 0.25}"
echo "      η ∈ {0.005, 0.01, 0.02}"
echo "Configs: λ_H ∈ {0.0, 1.0}"
echo "Total: 18 trainings × ${STEPS} steps"
echo "========================================="
echo

# Counter
count=0

# Loop over parameter grid
for beta in "${BETAS[@]}"; do
    for eta in "${ETAS[@]}"; do
        for lambda_h in "${LAMBDAS[@]}"; do
            count=$((count + 1))
            
            config_name="β=${beta}_η=${eta}_λ=${lambda_h}"
            echo "[$count/18] Launching: $config_name"
            
            # Launch in background with nohup
            nohup python3 train_parameter_sweep_v2.py \
                --beta "$beta" \
                --eta "$eta" \
                --lambda_h "$lambda_h" \
                --steps "$STEPS" \
                > "sweep_logs/sweep_beta${beta}_eta${eta}_lambda${lambda_h}.log" 2>&1 &
            
            pid=$!
            echo "  PID: $pid"
            echo "$pid" > "sweep_logs/sweep_beta${beta}_eta${eta}_lambda${lambda_h}.pid"
            
            # Small delay to avoid overwhelming system
            sleep 2
        done
    done
done

echo
echo "========================================="
echo "All 18 trainings launched!"
echo "========================================="
echo "Monitor progress:"
echo "  tail -f sweep_logs/*.log"
echo
echo "Check running processes:"
echo "  ps aux | grep train_parameter_sweep"
echo
echo "Estimated completion: 1-2 days"
echo "========================================="
