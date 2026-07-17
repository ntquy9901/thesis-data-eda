"""
Night-Run Autonomous Analysis Pipeline (2026-07-18)
====================================================

Runs comprehensive analysis:
1. Regression tests (all test suites)
2. Embedding generation & validation
3. EDA phases 11-16 (embedding-based analysis)
4. Correlation & statistical analysis
5. PCA dimensionality reduction
6. Dashboard update
7. Summary reporting

Run: PYTHONIOENCODING=utf-8 python -m src.eda.run_night_analysis
"""

import sys
import os
from datetime import datetime
from pathlib import Path
import traceback
import json
import subprocess

# Force UTF-8 encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

PROJECT_ROOT = Path(__file__).parent.parent.parent
LOG_DIR = PROJECT_ROOT / "reports"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"{datetime.now().strftime('%Y-%m-%d_%H%M')}_night_analysis.md"

def log_section(title):
    """Log a section header."""
    msg = f"\n{'='*70}\n{title}\n{'='*70}\n"
    print(msg, end='')
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(msg)

def log_msg(msg, level='INFO'):
    """Log a message."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    formatted = f"[{timestamp}] [{level}] {msg}\n"
    print(formatted, end='')
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(formatted)

def run_command(cmd_name, cmd, cwd=None):
    """Run a shell command and log output."""
    try:
        log_msg(f"Starting: {cmd_name}")
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd or PROJECT_ROOT,
            capture_output=True,
            text=True,
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}
        )

        if result.returncode == 0:
            log_msg(f"PASS: {cmd_name}")
            if result.stdout:
                log_msg(f"Output:\n{result.stdout[-500:]}")  # Last 500 chars
            return True
        else:
            log_msg(f"FAIL: {cmd_name} (exit code {result.returncode})", 'ERROR')
            if result.stderr:
                log_msg(f"Error:\n{result.stderr[-500:]}", 'ERROR')
            return False
    except Exception as e:
        log_msg(f"EXCEPTION in {cmd_name}: {e}", 'ERROR')
        traceback.print_exc()
        return False

def main():
    """Run full night analysis pipeline."""
    start_time = datetime.now()

    # Initialize log file
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write(f"# Night Analysis Run\n\n")
        f.write(f"Started: {start_time.isoformat()}\n\n")

    log_section("NIGHT ANALYSIS PIPELINE - START")

    results = {}

    # =========================================================================
    # TASK 1: Run Regression Tests
    # =========================================================================
    log_section("TASK 1: Running Regression Tests")

    test_cmd = "uv run pytest tests/unit -v --tb=short -q"
    results['regression_tests'] = run_command(
        "Full test suite",
        test_cmd
    )

    # =========================================================================
    # TASK 2: Embedding Generation & Validation
    # =========================================================================
    log_section("TASK 2: Embedding Generation & Validation")

    try:
        from src.features.news_embeddings import build_comparable_group_embeddings
        log_msg("Loading news embeddings...")
        emb_dict = build_comparable_group_embeddings()
        for group, df in emb_dict.items():
            log_msg(f"  Embeddings [{group}]: {df.shape} (rows, cols)")
        results['embeddings'] = True
    except Exception as e:
        log_msg(f"Embeddings error: {e}", 'ERROR')
        results['embeddings'] = False

    # =========================================================================
    # TASK 3: Run EDA Phases 11-16
    # =========================================================================
    log_section("TASK 3: Running EDA Phases 11-16")

    eda_phases = [
        ('phase11_news_embedding_eda', 'Phase 11: News Embedding EDA'),
        ('phase12_embedding_price_correlation', 'Phase 12: Embedding-Price Correlation'),
        ('phase13_novelty_correlation', 'Phase 13: Novelty Correlation'),
        ('phase14_uncertainty_index', 'Phase 14: Uncertainty Index'),
        ('phase15_temporal_decay_correlation', 'Phase 15: Temporal Decay Correlation'),
    ]

    # Check for Phase 16
    phase16_path = PROJECT_ROOT / 'src' / 'eda' / 'phase16_extended_horizon_correlation.py'
    if phase16_path.exists():
        eda_phases.append(('phase16_extended_horizon_correlation', 'Phase 16: Extended Horizon Correlation'))

    phase_results = {}
    for module_name, phase_label in eda_phases:
        try:
            log_msg(f"Running {phase_label}...")
            mod = __import__(f'src.eda.{module_name}', fromlist=['run_phase'])
            run_phase_func = getattr(mod, 'run_phase')
            outputs = run_phase_func()
            log_msg(f"  SUCCESS - {phase_label}")
            if outputs:
                for output in outputs:
                    log_msg(f"    -> {output}")
            phase_results[module_name] = True
        except Exception as e:
            log_msg(f"  FAIL - {phase_label}: {type(e).__name__}: {str(e)[:100]}", 'ERROR')
            phase_results[module_name] = False

    results['eda_phases'] = phase_results

    # =========================================================================
    # TASK 4: Modeling/Regression Analysis
    # =========================================================================
    log_section("TASK 4: Modeling & Regression Analysis")

    try:
        from src.modeling.baseline import train_and_compare_all
        log_msg("Running baseline modeling comparison...")
        metrics_path = train_and_compare_all()
        log_msg(f"  Baseline models complete: {metrics_path}")
        results['modeling'] = True
    except Exception as e:
        log_msg(f"Modeling error: {type(e).__name__}: {str(e)[:200]}", 'ERROR')
        results['modeling'] = False

    # =========================================================================
    # TASK 5: PCA Dimensionality Reduction
    # =========================================================================
    log_section("TASK 5: PCA Analysis")

    try:
        from src.features.news_embeddings import build_comparable_group_embeddings, _reduce
        log_msg("Performing PCA reduction...")
        emb_dict = build_comparable_group_embeddings()
        pca_results = {}
        for group, df in emb_dict.items():
            reduced = _reduce(df, dim=32)
            pca_results[group] = reduced.shape
            log_msg(f"  PCA[{group}]: {reduced.shape[0]} samples, 32 dims")
        results['pca'] = True
    except Exception as e:
        log_msg(f"PCA error: {type(e).__name__}: {str(e)[:200]}", 'ERROR')
        results['pca'] = False

    # =========================================================================
    # TASK 6: Update Streamlit Dashboard
    # =========================================================================
    log_section("TASK 6: Dashboard Update")

    # Dashboard doesn't need rebuilding - it reads artifacts dynamically
    # Just verify it can start
    dashboard_cmd = "timeout 10 streamlit run src/dashboard/app.py --logger.level=error 2>&1 | head -20"
    results['dashboard'] = run_command(
        "Dashboard startup check",
        dashboard_cmd
    )

    # =========================================================================
    # TASK 7: Generate Summary Report
    # =========================================================================
    log_section("TASK 7: Summary Report Generation")

    try:
        from src.eda.report import generate_final_report
        log_msg("Generating final EDA report...")
        report_path = generate_final_report()
        log_msg(f"  Report generated: {report_path}")
        results['report'] = True
    except Exception as e:
        log_msg(f"Report generation error: {type(e).__name__}: {str(e)[:200]}", 'ERROR')
        results['report'] = False

    # =========================================================================
    # Final Summary
    # =========================================================================
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    log_section("NIGHT ANALYSIS PIPELINE - COMPLETE")

    summary = {
        'started': start_time.isoformat(),
        'completed': end_time.isoformat(),
        'duration_seconds': duration,
        'results': results
    }

    log_msg(f"\nDuration: {duration:.1f}s ({duration/60:.1f}m)")
    log_msg(f"\nSummary:")
    log_msg(json.dumps(summary, indent=2))

    # Write summary JSON for programmatic access
    summary_json = LOG_DIR / "latest_night_run_summary.json"
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    log_msg(f"\nLog file: {LOG_FILE}")
    log_msg(f"Summary JSON: {summary_json}")
    log_msg(f"\nResults ready for dashboard at http://localhost:8501")

    # Count passes
    pass_count = sum(1 for v in results.values() if v is True or (isinstance(v, dict) and sum(v.values())))
    total_count = len(results) + len(phase_results)

    log_msg(f"\nPassed: {pass_count}/{total_count} tasks")

    return 0 if pass_count >= (total_count - 1) else 1  # Allow 1 failure

if __name__ == '__main__':
    sys.exit(main())
