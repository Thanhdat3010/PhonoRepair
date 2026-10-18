"""LaTeX and Markdown export utilities for Interspeech Table 1 and Table 2."""

from typing import Dict, List, Optional


def export_table_1_markdown(results: List[Dict[str, any]]) -> str:
    """Format evaluation results into Table 1 Markdown table."""
    lines = [
        "| Backbone | Method | LibriSpeech (Clean) WER (%) | UASpeech WER (%) | UASpeech Rel. WERR (%) | TORGO WER (%) | TORGO Rel. WERR (%) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for row in results:
        sig_ua = "*" if row.get("uaspeech_sig", False) else ""
        sig_torgo = "*" if row.get("torgo_sig", False) else ""
        lines.append(
            f"| **{row['backbone']}** | {row['method']} | {row['clean_wer']:.2f} | "
            f"{row['uaspeech_wer']:.2f} | {row['uaspeech_werr']:+.1f}%{sig_ua} | "
            f"{row['torgo_wer']:.2f} | {row['torgo_werr']:+.1f}%{sig_torgo} |"
        )

    return "\n".join(lines)


def export_table_1_latex(results: List[Dict[str, any]]) -> str:
    """Format evaluation results into camera-ready Interspeech LaTeX table."""
    latex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Main recognition performance across frozen ASR backbones on clean and dysarthric speech benchmarks. * denotes statistical significance ($p < 0.05$ under paired cluster bootstrap).}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"\textbf{Backbone} & \textbf{Method} & \textbf{LibriSpeech (Clean) $\downarrow$} & \textbf{UASpeech $\downarrow$} & \textbf{UASpeech WERR $\uparrow$} & \textbf{TORGO $\downarrow$} & \textbf{TORGO WERR $\uparrow$} \\",
        r"\midrule",
    ]

    current_bb = None
    for row in results:
        bb = row["backbone"]
        bb_col = f"\\textbf{{{bb}}}" if bb != current_bb else ""
        current_bb = bb

        sig_ua = r"$^*$" if row.get("uaspeech_sig", False) else ""
        sig_torgo = r"$^*$" if row.get("torgo_sig", False) else ""

        werr_ua_str = f"{row['uaspeech_werr']:+.1f}\\%{sig_ua}" if row['uaspeech_werr'] != 0 else "--"
        werr_tor_str = f"{row['torgo_werr']:+.1f}\\%{sig_torgo}" if row['torgo_werr'] != 0 else "--"

        latex.append(
            f"{bb_col} & {row['method']} & {row['clean_wer']:.2f} & {row['uaspeech_wer']:.2f} & {werr_ua_str} & {row['torgo_wer']:.2f} & {werr_tor_str} \\\\"
        )

    latex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    return "\n".join(latex)
