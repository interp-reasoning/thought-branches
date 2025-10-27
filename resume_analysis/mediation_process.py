import sys
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats

# Suppress warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ProcessModel4:
    """Process Model 4 - Parallel Multiple Mediation Analysis"""

    def __init__(self, data, x, y, m, boot=5000, seed=42):
        self.data = data
        self.x = x
        self.y = y
        self.m = m if isinstance(m, list) else [m]
        self.boot = boot
        self.seed = seed
        np.random.seed(seed)

        # Run analysis
        self._run_analysis()

    def _run_analysis(self):
        """Run the mediation analysis"""
        X = self.data[self.x].values.reshape(-1, 1)
        Y = self.data[self.y].values
        M = self.data[self.m].values

        # Total effect (c path)
        X_with_const = sm.add_constant(X)
        model_total = sm.OLS(Y, X_with_const).fit()
        self.total_effect = model_total.params[1]
        self.total_se = model_total.bse[1]
        self.total_p = model_total.pvalues[1]
        self.total_ci = model_total.conf_int()[1]

        # Direct effect (c' path) - controlling for all mediators
        XM = np.hstack([X, M])
        XM_with_const = sm.add_constant(XM)
        model_direct = sm.OLS(Y, XM_with_const).fit()
        self.direct_effect = model_direct.params[1]
        self.direct_se = model_direct.bse[1]
        self.direct_p = model_direct.pvalues[1]
        self.direct_ci = model_direct.conf_int()[1]

        # Calculate indirect effect
        self.indirect_effect = self.total_effect - self.direct_effect

        # Bootstrap for indirect effect CI
        self._bootstrap_indirect()

    def _bootstrap_indirect(self):
        """Bootstrap confidence intervals for indirect effect"""
        n = len(self.data)
        boot_indirect = []

        X = self.data[self.x].values.reshape(-1, 1)
        Y = self.data[self.y].values
        M = self.data[self.m].values

        for _ in range(self.boot):
            idx = np.random.choice(n, n, replace=True)
            X_b = X[idx]
            Y_b = Y[idx]
            M_b = M[idx]

            # Bootstrap total effect
            X_b_const = sm.add_constant(X_b)
            try:
                model_t = sm.OLS(Y_b, X_b_const).fit()
                c_b = model_t.params[1]

                # Bootstrap direct effect
                XM_b = np.hstack([X_b, M_b])
                XM_b_const = sm.add_constant(XM_b)
                model_d = sm.OLS(Y_b, XM_b_const).fit()
                c_prime_b = model_d.params[1]

                boot_indirect.append(c_b - c_prime_b)
            except:
                continue

        # Calculate CI
        self.indirect_ci = np.percentile(boot_indirect, [2.5, 97.5])

    def summary(self):
        """Print summary of results"""
        print("\n" + "=" * 70)
        print("PROCESS MODEL 4 - PARALLEL MULTIPLE MEDIATION")
        print("=" * 70)

        print(f"\nTotal Effect of {self.x} on {self.y}:")
        print(f"  Coefficient: {self.total_effect:.4f}")
        print(f"  SE: {self.total_se:.4f}")
        print(f"  95% CI: [{self.total_ci[0]:.4f}, {self.total_ci[1]:.4f}]")
        print(f"  p-value: {self.total_p:.4f}")

        print(f"\nDirect Effect of {self.x} on {self.y} (controlling for mediators):")
        print(f"  Coefficient: {self.direct_effect:.4f}")
        print(f"  SE: {self.direct_se:.4f}")
        print(f"  95% CI: [{self.direct_ci[0]:.4f}, {self.direct_ci[1]:.4f}]")
        print(f"  p-value: {self.direct_p:.4f}")

        print(f"\nIndirect Effect of {self.x} on {self.y} (through mediators):")
        print(f"  Coefficient: {self.indirect_effect:.4f}")
        print(f"  95% Boot CI: [{self.indirect_ci[0]:.4f}, {self.indirect_ci[1]:.4f}]")

        is_sig = self.indirect_ci[0] > 0 or self.indirect_ci[1] < 0
        print(f"  Significant: {'Yes' if is_sig else 'No'}")

        if self.total_effect != 0:
            prop_mediated = abs(self.indirect_effect / self.total_effect * 100)
            print(f"  Proportion Mediated: {prop_mediated:.1f}%")

        print("=" * 70)

if __name__ == "__main__":
    N_CLUSTERS = 16
    INCLUDE_MIRROR = True
    JOB = "basic"
    RESUME_IDS = [13, 39, 0, 7, 69, 73, 83, 61]

    print("MEDIATION ANALYSIS USING PROCESS-STYLE MODEL 4")
    print("=" * 70)

    results_all = {}

    for RESUME_ID in RESUME_IDS:
        print(f"\nProcessing resume: {RESUME_ID}")
        print("-" * 60)

        try:
            # Load data
            df = pd.read_csv(
                rf"bias_csvs/sentence_{JOB}_{N_CLUSTERS}_{INCLUDE_MIRROR}_{RESUME_ID}.csv"
            )

            # Filter to variants 1 and 2
            df = df[df["variant"].isin({1, 2})]

            # Define mediator columns
            mediator_columns = [f"x_{i}" for i in range(0, 16)]

            # Check for and remove NaN values
            columns_to_check = ['decision', 'variant'] + mediator_columns
            df_clean = df.dropna(subset=columns_to_check)

            # Check if we have enough data
            if len(df_clean) < 50:
                print(f"  Warning: Only {len(df_clean)} valid rows after filtering NaN values")
                continue

            print(f"  Data points: {len(df_clean)}")

            # Run Process Model 4 analysis
            process = ProcessModel4(
                data=df_clean,
                x="variant",
                y="decision",
                m=mediator_columns,
                boot=5000,  # Number of bootstrap samples
                seed=42,  # Random seed for reproducibility
            )

            # Store results
            results = {
                'direct_effect': process.direct_effect,
                'direct_llci': process.direct_ci[0],
                'direct_ulci': process.direct_ci[1],
                'direct_p': process.direct_p,
                'indirect_effect': process.indirect_effect,
                'indirect_llci': process.indirect_ci[0],
                'indirect_ulci': process.indirect_ci[1],
                'total_effect': process.total_effect,
            }

            results_all[RESUME_ID] = results

            # Print results
            print("\n  RESULTS:")
            print(f"  Direct Effect (variant → decision):")
            print(f"    Coefficient: {results['direct_effect']:.4f}")
            print(f"    95% CI: [{results['direct_llci']:.4f}, {results['direct_ulci']:.4f}]")
            print(f"    p-value: {results['direct_p']:.4f}")

            print(f"\n  Total Indirect Effect (via all mediators):")
            print(f"    Coefficient: {results['indirect_effect']:.4f}")
            print(f"    95% Boot CI: [{results['indirect_llci']:.4f}, {results['indirect_ulci']:.4f}]")

            # Check significance
            is_sig = (results['indirect_llci'] > 0 or results['indirect_ulci'] < 0)
            print(f"    Significant: {'Yes' if is_sig else 'No'}")

            print(f"\n  Total Effect: {results['total_effect']:.4f}")

            # Calculate proportion mediated
            if results['total_effect'] != 0:
                prop_med = abs(results['indirect_effect'] / results['total_effect'] * 100)
                print(f"  Proportion Mediated: {prop_med:.1f}%")

            print("-" * 60)

        except Exception as e:
            print(f"  Error processing resume {RESUME_ID}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY ACROSS ALL RESUMES:")
    print("=" * 70)

    for resume_id, res in results_all.items():
        # Check indirect effect significance
        sig_indirect = "✓" if (res['indirect_llci'] > 0 or res['indirect_ulci'] < 0) else "✗"

        print(f"Resume {resume_id}:")
        print(f"  Direct Effect: {res['direct_effect']:.4f}")
        print(f"  Indirect Effect: {res['indirect_effect']:.4f} [{sig_indirect}]")
        print(f"  Total Effect: {res['total_effect']:.4f}")
        print()