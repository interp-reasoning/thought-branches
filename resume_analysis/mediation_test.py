import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

def mediation_analysis(df, x_var, y_var, mediators, n_boot=10000):
    """Clean mediation analysis without pyprocessmacro"""
    X = df[x_var].values.reshape(-1, 1)
    Y = df[y_var].values
    M = df[mediators].values
    
    # Total effect (c path)
    model_total = LinearRegression()
    model_total.fit(X, Y)
    total_effect = model_total.coef_[0]
    
    # Direct effect (c' path) - controlling for all mediators
    X_with_M = np.hstack([X, M])
    model_direct = LinearRegression()
    model_direct.fit(X_with_M, Y)
    direct_effect = model_direct.coef_[0]
    
    # Calculate p-value for direct effect using statsmodels
    import statsmodels.api as sm
    X_with_M_const = sm.add_constant(X_with_M)
    model_sm = sm.OLS(Y, X_with_M_const).fit()
    direct_p = model_sm.pvalues[1]  # p-value for X
    direct_ci = model_sm.conf_int()[1]  # CI for X
    
    # Indirect effect
    indirect_effect = total_effect - direct_effect
    
    # Bootstrap for indirect effect CI
    n = len(X)
    boot_indirect = []
    
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        X_b = X[idx]
        Y_b = Y[idx]
        M_b = M[idx]
        
        # Bootstrap total effect
        model_t = LinearRegression()
        model_t.fit(X_b, Y_b)
        c_b = model_t.coef_[0]
        
        # Bootstrap direct effect
        XM_b = np.hstack([X_b, M_b])
        model_d = LinearRegression()
        model_d.fit(XM_b, Y_b)
        c_prime_b = model_d.coef_[0]
        
        boot_indirect.append(c_b - c_prime_b)
    
    # Calculate CI
    indirect_ci = np.percentile(boot_indirect, [2.5, 97.5])
    
    return {
        'direct_effect': direct_effect,
        'direct_p': direct_p,
        'direct_ci': direct_ci,
        'indirect_effect': indirect_effect,
        'indirect_ci': indirect_ci,
        'total_effect': total_effect
    }

# Main loop
if __name__ == "__main__":
    N_CLUSTERS = 16
    INCLUDE_MIRROR = True
    JOB = "basic"
    RESUME_IDS = [13, 39, 0, 7, 69, 73, 83, 61]
    
    results_all = {}
    
    for RESUME_ID in RESUME_IDS:
        print(f"\nProcessing resume: {RESUME_ID}")
        
        df = pd.read_csv(
            rf"bias_csvs/sentence_{JOB}_{N_CLUSTERS}_{INCLUDE_MIRROR}_{RESUME_ID}.csv"
        )
        df = df[df["variant"].isin({1, 2})]

        # Remove rows with NaN values in decision or mediators
        mediator_columns = [f"x_{i}" for i in range(0, 16)]
        columns_to_check = ['decision', 'variant'] + mediator_columns
        df = df.dropna(subset=columns_to_check)

        # Check if we have enough data
        if len(df) < 50:
            print(f"  Warning: Only {len(df)} valid rows after filtering NaN values")
            continue

        # Run analysis
        results = mediation_analysis(df, 'variant', 'decision', mediator_columns)
        results_all[RESUME_ID] = results
        
        # Print results
        print("=" * 60)
        print(f"MEDIATION ANALYSIS RESULTS - Resume {RESUME_ID}")
        print("=" * 60)
        print(f"\nDirect Effect (variant → decision):")
        print(f"  Effect: {results['direct_effect']:.4f}")
        print(f"  95% CI: [{results['direct_ci'][0]:.4f}, {results['direct_ci'][1]:.4f}]")
        print(f"  p-value: {results['direct_p']:.4f}")
        
        print(f"\nTotal Indirect Effect (via all mediators):")
        print(f"  Effect: {results['indirect_effect']:.4f}")
        print(f"  95% CI: [{results['indirect_ci'][0]:.4f}, {results['indirect_ci'][1]:.4f}]")
        
        is_sig = "Yes" if (results['indirect_ci'][0] > 0 or results['indirect_ci'][1] < 0) else "No"
        print(f"  Significant: {is_sig}")
        
        # Additional stats
        total = results['total_effect']
        prop_med = abs(results['indirect_effect'] / total * 100) if total != 0 else 0
        print(f"\nAdditional Information:")
        print(f"  Total Effect: {total:.4f}")
        print(f"  Proportion Mediated: {prop_med:.1f}%")
        print("=" * 60)
    
    # Summary across all resumes
    print("\n\nSUMMARY ACROSS ALL RESUMES:")
    print("=" * 60)
    for resume_id, res in results_all.items():
        sig = "✓" if (res['indirect_ci'][0] > 0 or res['indirect_ci'][1] < 0) else "✗"
        print(f"Resume {resume_id}: Indirect={res['indirect_effect']:.4f} [{sig}], Direct={res['direct_effect']:.4f}")