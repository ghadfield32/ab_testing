"""
Run All A/B Testing Pipelines

This script demonstrates how to run the complete end-to-end A/B testing pipeline
on all three real-world datasets.

Usage:
    # Run all pipelines
    uv run python run_pipelines.py

    # Run specific pipeline
    uv run python run_pipelines.py --pipeline marketing

    # Run with custom sample size
    uv run python run_pipelines.py --pipeline criteo --sample 0.001

    # Run quietly
    uv run python run_pipelines.py --quiet
"""

import argparse
import sys
from typing import Optional

from ab_testing.pipelines import (
    run_marketing_analysis,
    run_criteo_analysis,
    run_cookie_cats_analysis
)


def run_all_pipelines(verbose: bool = True):
    """Run all three pipeline demonstrations."""

    if verbose:
        print("\n" + "="*80)
        print(" "*20 + "A/B TESTING PIPELINE DEMONSTRATIONS")
        print("="*80)
        print("\nThis will run complete A/B test analyses on three real-world datasets:")
        print("  1. Marketing A/B Test (588K observations)")
        print("  2. Criteo Uplift (13.9M observations - sampled)")
        print("  3. Cookie Cats Mobile Game (90K players)")
        print("\n" + "="*80 + "\n")

    results = {}

    # Pipeline 1: Marketing A/B Test
    try:
        if verbose:
            print("\n" + "█"*80)
            print("PIPELINE 1/3: MARKETING A/B TEST")
            print("█"*80 + "\n")

        results['marketing'] = run_marketing_analysis(sample_frac=0.1, verbose=verbose)

        if verbose:
            print("\n✓ Marketing pipeline completed successfully\n")
    except Exception as e:
        print(f"\n✗ Marketing pipeline failed: {e}\n")
        if not verbose:
            raise

    # Pipeline 2: Criteo Uplift
    try:
        if verbose:
            print("\n" + "█"*80)
            print("PIPELINE 2/3: CRITEO UPLIFT MODELING")
            print("█"*80 + "\n")

        results['criteo'] = run_criteo_analysis(sample_frac=0.001, verbose=verbose)

        if verbose:
            print("\n✓ Criteo pipeline completed successfully\n")
    except Exception as e:
        print(f"\n✗ Criteo pipeline failed: {e}\n")
        if not verbose:
            raise

    # Pipeline 3: Cookie Cats
    try:
        if verbose:
            print("\n" + "█"*80)
            print("PIPELINE 3/3: COOKIE CATS MOBILE GAME")
            print("█"*80 + "\n")

        results['cookie_cats'] = run_cookie_cats_analysis(verbose=verbose)

        if verbose:
            print("\n✓ Cookie Cats pipeline completed successfully\n")
    except Exception as e:
        print(f"\n✗ Cookie Cats pipeline failed: {e}\n")
        if not verbose:
            raise

    # Summary
    if verbose:
        print("\n" + "="*80)
        print(" "*30 + "FINAL SUMMARY")
        print("="*80)

        for name, result in results.items():
            decision = result.get('decision', {}).get('decision', 'N/A').upper()
            print(f"\n{name.upper()}: Decision = {decision}")

            if 'primary_test' in result:
                p_val = result['primary_test'].get('p_value', -1)
                sig = result['primary_test'].get('significant', False)
                print(f"  P-value: {p_val:.6f} | Significant: {sig}")

        print("\n" + "="*80)
        print("All pipelines completed!")
        print("="*80 + "\n")

    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run A/B testing pipeline demonstrations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all pipelines
  python run_pipelines.py

  # Run specific pipeline
  python run_pipelines.py --pipeline marketing
  python run_pipelines.py --pipeline criteo
  python run_pipelines.py --pipeline cookie_cats

  # Run with custom sample size
  python run_pipelines.py --pipeline criteo --sample 0.01

  # Run quietly (no output)
  python run_pipelines.py --quiet
        """
    )

    parser.add_argument(
        '--pipeline',
        choices=['all', 'marketing', 'criteo', 'cookie_cats'],
        default='all',
        help='Which pipeline to run (default: all)'
    )

    parser.add_argument(
        '--sample',
        type=float,
        default=None,
        help='Sample fraction for data loading (0.0-1.0)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )

    args = parser.parse_args()

    verbose = not args.quiet

    try:
        if args.pipeline == 'all':
            run_all_pipelines(verbose=verbose)

        elif args.pipeline == 'marketing':
            sample_frac = args.sample if args.sample is not None else 0.1
            result = run_marketing_analysis(sample_frac=sample_frac, verbose=verbose)
            if verbose:
                print(f"\nDecision: {result['decision']['decision'].upper()}")

        elif args.pipeline == 'criteo':
            sample_frac = args.sample if args.sample is not None else 0.001
            result = run_criteo_analysis(sample_frac=sample_frac, verbose=verbose)
            if verbose:
                print(f"\nDecision: {result['decision']['decision'].upper()}")

        elif args.pipeline == 'cookie_cats':
            sample_frac = args.sample if args.sample is not None else 1.0
            result = run_cookie_cats_analysis(sample_frac=sample_frac, verbose=verbose)
            if verbose:
                print(f"\nDecision: {result['decision']['decision'].upper()}")

        return 0

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
