import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyprocessmacro import Process
import pandas as pd
import numpy as np

if __name__ == "__main__":
    N_CLUSTERS = 16
    INCLUDE_MIRROR = True
    JOB = "basic"
    RESUME_IDS = [13, 39, 0, 7, 69, 73, 83]
    
    print("TEST")
    
    for i, RESUME_ID in enumerate(RESUME_IDS):
        print(f"\nProcessing resume: {RESUME_ID}")
        df = pd.read_csv(
            rf"bias_csvs/sentence_{JOB}_{N_CLUSTERS}_{INCLUDE_MIRROR}_{RESUME_ID}.csv"
        )
        df = df[df["variant"].isin({1, 2})]
        
        mediator_columns = [f"x_{i}" for i in range(0, 16)]
        
        # Run Process silently
        process = Process(
            data=df,
            model=4,
            x="variant",
            y="decision",
            m=mediator_columns,
            total=1,
            contrast=1,
            boot=10_000,
            seed=42,
            suppr_init=True,
            detail=False  # This reduces output
        )
        
        # Access the data structures directly
        # The Process object stores everything internally
        
        # Try to find the right attributes
        # Direct effect info
        print
        direct_results = process.models['outcome']['results']
        direct_coeff = direct_results.params[1]  # Usually the second parameter
        direct_pvalue = direct_results.pvalues[1]
        direct_ci = direct_results.conf_int()[1]
        
        # For indirect effects, check these attributes
        if hasattr(process, 'boot_indirect_effect'):
            indirect_effect = process.boot_indirect_effect[0]  # Total is usually first
            indirect_ci = process.boot_indirect_effect_ci[0]
        else:
            # Alternative: manually calculate
            total_effect = process.models['total']['results'].params[1]
            indirect_effect = total_effect - direct_coeff
            # Bootstrap CI would need to be calculated
            indirect_ci = (np.nan, np.nan)
        
        # Print results
        print("=" * 60)
        print(f"MEDIATION ANALYSIS RESULTS - Resume {RESUME_ID}")
        print("=" * 60)
        print(f"\nDirect Effect (variant → decision):")
        print(f"  Effect: {direct_coeff:.4f}")
        print(f"  95% CI: [{direct_ci[0]:.4f}, {direct_ci[1]:.4f}]")
        print(f"  p-value: {direct_pvalue:.4f}")
        
        print(f"\nTotal Indirect Effect (via all mediators):")
        print(f"  Effect: {indirect_effect:.4f}")
        if not np.isnan(indirect_ci[0]):
            print(f"  95% CI: [{indirect_ci[0]:.4f}, {indirect_ci[1]:.4f}]")
            is_sig = "Yes" if (indirect_ci[0] > 0 or indirect_ci[1] < 0) else "No"
            print(f"  Significant: {is_sig}")
        
        print("=" * 60)