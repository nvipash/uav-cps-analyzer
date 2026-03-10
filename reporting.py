#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UAV-CPS-Analyzer: Reporting Module
LaTeX table generation and comprehensive statistical summary reports.

Authors: Novitskyi P.S., Stepaniak M.V.
Lviv Polytechnic National University, 2025
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

try:
    from monte_carlo_engine import MCResult
    from sensitivity import SobolResult, MorrisResult
    from validation import ValidationReport
except ImportError:
    import sys
    sys.path.insert(0, '.')
    from monte_carlo_engine import MCResult
    from sensitivity import SobolResult, MorrisResult
    from validation import ValidationReport


def _round_sig(x: float, sig: int = 3) -> str:
    """Round to significant figures."""
    if x == 0:
        return "0"
    from math import log10, floor
    digits = sig - int(floor(log10(abs(x)))) - 1
    return f"{x:.{max(0, digits)}f}"


def generate_latex_table(headers: List[str], rows: List[List[str]],
                         caption: str = "", label: str = "",
                         notes: str = "") -> str:
    """
    Generate a complete LaTeX table with booktabs formatting.

    Args:
        headers: Column header strings
        rows: List of row data (list of strings)
        caption: Table caption
        label: LaTeX label for cross-referencing
        notes: Table footnotes

    Returns:
        Complete LaTeX table string
    """
    n_cols = len(headers)
    col_spec = "l" + "r" * (n_cols - 1)

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}" if caption else "",
        f"\\label{{{label}}}" if label else "",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]

    for row in rows:
        lines.append(" & ".join(row) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
    ])

    if notes:
        lines.append(f"\\\\[2pt]")
        lines.append(f"\\footnotesize{{{notes}}}")

    lines.append("\\end{table}")

    return "\n".join(line for line in lines if line)


def table3_latex(scenario_results: Dict[str, MCResult]) -> str:
    """
    Generate LaTeX for Table 3: Monte Carlo Simulation Results.

    Args:
        scenario_results: Dict of scenario_name -> MCResult

    Returns:
        LaTeX table string
    """
    headers = ["Scenario", "$\\bar{J/S}$ (dB)", "$\\sigma$ (dB)",
               "95\\% CI (dB)", "$P_{\\text{jam}}$", "Converged", "N"]

    rows = []
    for name, result in scenario_results.items():
        ci_str = f"[{result.ci_95_lower:.1f}, {result.ci_95_upper:.1f}]"
        conv_str = "Yes" if result.converged else "No"
        rows.append([
            name.replace('_', ' ').title(),
            f"{result.mean_js_db:.1f}",
            f"{result.std_js_db:.1f}",
            ci_str,
            f"{result.success_probability*100:.0f}\\%",
            conv_str,
            f"{result.n_iterations:,}"
        ])

    return generate_latex_table(
        headers, rows,
        caption="Monte Carlo simulation results for jamming scenarios",
        label="tab:mc_results",
        notes="CI computed via percentile method with bootstrap verification."
    )


def table4_latex(fhss_results: Dict[str, dict]) -> str:
    """
    Generate LaTeX for Table 4: FHSS Strategy Effectiveness.

    Args:
        fhss_results: Dict from JammingEffectivenessAnalyzer.compare_strategies()

    Returns:
        LaTeX table string
    """
    headers = ["Strategy", "Static (\\%)", "FHSS (\\%)", "Power Mult. ($\\times$)"]

    rows = []
    for strategy, data in fhss_results.items():
        rows.append([
            strategy.replace('_', ' ').title(),
            f"{data['effectiveness_static']*100:.0f}",
            f"{data['effectiveness_fhss']*100:.0f}",
            f"{data['power_multiplier']:.0f}"
        ])

    return generate_latex_table(
        headers, rows,
        caption="Jamming strategy effectiveness: static channel vs. FHSS (OcuSync 3.0)",
        label="tab:fhss_effectiveness"
    )


def table7_latex(validation_report: ValidationReport) -> str:
    """
    Generate LaTeX for Table 7: Model Validation.

    Args:
        validation_report: ValidationReport from ValidationEngine

    Returns:
        LaTeX table string
    """
    headers = ["Scenario", "Model (dB)", "95\\% CI", "Reference",
               "Error (dB)", "MAPE", "V\\&V 20"]

    rows = []
    for case in validation_report.cases:
        if case.params is None:
            continue
        ci_str = f"[{case.model_ci[0]:.1f}, {case.model_ci[1]:.1f}]"
        ref_str = f"{case.reference_value:.1f}"
        vv_str = "Pass" if case.passed_vv20 else "Fail"
        rows.append([
            case.name,
            f"{case.model_prediction:.1f}",
            ci_str,
            ref_str,
            f"{case.model_error:.2f}",
            f"{case.mape:.1f}\\%",
            vv_str
        ])

    return generate_latex_table(
        headers, rows,
        caption="Model validation against experimental references (ASME V\\&V 20)",
        label="tab:validation",
        notes=f"Overall MAPE: {validation_report.overall_mape:.1f}\\%, "
              f"V\\&V 20 pass rate: {validation_report.overall_vv20_pass_rate*100:.0f}\\%"
    )


def sobol_latex(sobol_result: SobolResult) -> str:
    """
    Generate LaTeX for Sobol sensitivity indices table.

    Args:
        sobol_result: SobolResult from sobol_analysis()

    Returns:
        LaTeX table string
    """
    headers = ["Parameter", "$S_1$", "$S_1$ CI", "$S_T$", "$S_T$ CI", "Interaction"]

    rows = []
    sorted_params = sorted(sobol_result.parameter_names,
                           key=lambda p: sobol_result.ST[p], reverse=True)
    for name in sorted_params:
        rows.append([
            name.replace('_', ' ').title(),
            f"{sobol_result.S1[name]:.3f}",
            f"$\\pm${sobol_result.S1_conf[name]:.3f}",
            f"{sobol_result.ST[name]:.3f}",
            f"$\\pm${sobol_result.ST_conf[name]:.3f}",
            f"{sobol_result.interaction[name]:.3f}"
        ])

    return generate_latex_table(
        headers, rows,
        caption=f"Sobol sensitivity indices (N={sobol_result.n_samples}, "
                f"{sobol_result.total_model_evaluations} evaluations)",
        label="tab:sobol",
        notes="$S_1$: first-order (main effect), $S_T$: total-order, "
              "Interaction = $S_T - S_1$."
    )


def generate_summary_report(scenario_results: Dict[str, MCResult] = None,
                            fhss_results: Dict[str, dict] = None,
                            validation_report: ValidationReport = None,
                            sobol_result: SobolResult = None,
                            output_path: str = None) -> str:
    """
    Generate comprehensive statistical summary report in Markdown.

    Args:
        scenario_results: MC simulation results
        fhss_results: FHSS analysis results
        validation_report: Validation report
        sobol_result: Sobol analysis results
        output_path: Path to save report (optional)

    Returns:
        Report string in Markdown format
    """
    lines = [
        "# UAV-CPS-Analyzer: Statistical Summary Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Software version: 1.0.0",
        "",
    ]

    # MC Results
    if scenario_results:
        lines.extend([
            "## Monte Carlo Simulation Results",
            "",
            "| Scenario | Mean J/S (dB) | Std (dB) | 95% CI (dB) | P(jam) | Converged | N |",
            "|----------|--------------|---------|-------------|--------|-----------|---|",
        ])
        for name, r in scenario_results.items():
            ci = f"[{r.ci_95_lower:.1f}, {r.ci_95_upper:.1f}]"
            conv = "Yes" if r.converged else "No"
            lines.append(
                f"| {name} | {r.mean_js_db:.1f} | {r.std_js_db:.1f} | {ci} "
                f"| {r.success_probability*100:.0f}% | {conv} | {r.n_iterations:,} |"
            )

        # Bootstrap CI details
        lines.append("")
        lines.append("### Bootstrap CI Uncertainty")
        for name, r in scenario_results.items():
            if r.ci_lower_uncertainty != (0.0, 0.0):
                lines.append(
                    f"- **{name}**: CI lower [{r.ci_lower_uncertainty[0]:.2f}, "
                    f"{r.ci_lower_uncertainty[1]:.2f}], "
                    f"CI upper [{r.ci_upper_uncertainty[0]:.2f}, "
                    f"{r.ci_upper_uncertainty[1]:.2f}]"
                )

        # Sample size justification
        lines.append("")
        lines.append("### Sample Size Justification")
        for name, r in scenario_results.items():
            lines.append(
                f"- **{name}**: Required N = {r.required_sample_size:,} "
                f"(for 0.1 dB precision at 95% confidence), used N = {r.n_iterations:,}"
            )
        lines.append("")

    # FHSS Results
    if fhss_results:
        lines.extend([
            "## FHSS Effectiveness Analysis",
            "",
            "| Strategy | Static | FHSS | Power Multiplier |",
            "|----------|--------|------|-----------------|",
        ])
        for strategy, data in fhss_results.items():
            lines.append(
                f"| {strategy} | {data['effectiveness_static']*100:.0f}% "
                f"| {data['effectiveness_fhss']*100:.0f}% "
                f"| x{data['power_multiplier']:.0f} |"
            )
        lines.append("")

    # Sobol Results
    if sobol_result:
        lines.extend([
            "## Global Sensitivity Analysis (Sobol Indices)",
            f"Base sample size: N = {sobol_result.n_samples}, "
            f"total evaluations: {sobol_result.total_model_evaluations}",
            "",
            "| Parameter | S1 | S1 CI | ST | ST CI | Interaction |",
            "|-----------|-----|-------|-----|-------|------------|",
        ])
        sorted_p = sorted(sobol_result.parameter_names,
                          key=lambda p: sobol_result.ST[p], reverse=True)
        for name in sorted_p:
            lines.append(
                f"| {name} | {sobol_result.S1[name]:.3f} "
                f"| +/-{sobol_result.S1_conf[name]:.3f} "
                f"| {sobol_result.ST[name]:.3f} "
                f"| +/-{sobol_result.ST_conf[name]:.3f} "
                f"| {sobol_result.interaction[name]:.3f} |"
            )
        lines.append("")

    # Validation
    if validation_report:
        lines.extend([
            "## Model Validation (ASME V&V 20)",
            "",
            "| Case | Model (dB) | Reference (dB) | Error | MAPE | V&V 20 |",
            "|------|-----------|---------------|-------|------|--------|",
        ])
        for case in validation_report.cases:
            if case.params is None:
                continue
            vv = "PASS" if case.passed_vv20 else "FAIL"
            lines.append(
                f"| {case.name} | {case.model_prediction:.1f} "
                f"| {case.reference_value:.1f} "
                f"| {case.model_error:.2f} | {case.mape:.1f}% | {vv} |"
            )

        lines.extend([
            "",
            f"- Overall MAPE: {validation_report.overall_mape:.1f}%",
            f"- CI Coverage Rate: {validation_report.overall_coverage_rate*100:.0f}%",
            f"- V&V 20 Pass Rate: {validation_report.overall_vv20_pass_rate*100:.0f}%",
            f"- Total Cases: {validation_report.n_cases}",
            "",
        ])

        # Per-domain breakdown
        if validation_report.mape_by_domain:
            lines.extend([
                "### Per-Domain Validation Metrics",
                "",
                "| Domain | n | MAPE | V&V 20 PASS | CI Coverage |",
                "|--------|---|------|-------------|-------------|",
            ])
            for d in sorted(validation_report.n_by_domain.keys()):
                lines.append(
                    f"| {d} | {validation_report.n_by_domain[d]} "
                    f"| {validation_report.mape_by_domain[d]:.1f}% "
                    f"| {validation_report.pass_rate_by_domain[d]*100:.0f}% "
                    f"| {validation_report.coverage_by_domain[d]*100:.0f}% |"
                )
            lines.append("")

    # Footer
    lines.extend([
        "---",
        "Report generated by UAV-CPS-Analyzer v1.0.0",
        "Authors: Novitskyi P.S., Stepaniak M.V.",
        "Lviv Polytechnic National University, 2025",
    ])

    report = "\n".join(lines)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report saved to: {output_path}")

    return report


if __name__ == "__main__":
    print("=" * 60)
    print("UAV-CPS-Analyzer: Reporting Module Test")
    print("=" * 60)

    # Generate sample LaTeX table
    sample_result = MCResult(
        n_iterations=10000, mean_js_db=38.2, std_js_db=4.7,
        ci_95_lower=28.8, ci_95_upper=47.3, success_probability=1.0,
        converged=True, convergence_iteration=8500, required_sample_size=9604
    )

    latex = table3_latex({"portable_10W_500m": sample_result})
    print("\nSample LaTeX table:")
    print(latex)

    # Generate summary report
    report = generate_summary_report(
        scenario_results={"portable_10W_500m": sample_result}
    )
    print("\nSample summary report (first 30 lines):")
    for line in report.split('\n')[:30]:
        print(line)

    print("\n" + "=" * 60)
    print("Reporting test complete!")
