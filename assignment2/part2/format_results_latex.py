"""
Format generation results from JSON to LaTeX table.

Usage:
    python format_results_latex.py --input generation_results.json --output generation_table.tex
"""

import argparse
import json


def escape_latex(text):
    """Escape special LaTeX characters and handle newlines."""
    if text is None:
        return "[ERROR]"
    # Replace newlines with spaces for table formatting
    text = text.replace('\n', ' ').replace('\r', '')
    # Escape special characters (order matters - backslash first)
    text = text.replace('\\', '\\textbackslash{}')
    special_chars = ['&', '%', '$', '#', '_', '{', '}']
    for char in special_chars:
        text = text.replace(char, '\\' + char)
    text = text.replace('~', '\\textasciitilde{}')
    text = text.replace('^', '\\textasciicircum{}')
    return text


def truncate_text(text, max_chars=100):
    """Truncate text for display."""
    if text is None:
        return "[ERROR]"
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


def format_config_table(configurations):
    """Generate LaTeX table summarizing configurations."""
    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\caption{Summary of generation methods tested}")
    lines.append("\\begin{tabular}{|l|c|c|c|c|}")
    lines.append("\\hline")
    lines.append("\\textbf{Method} & \\textbf{Sampling} & \\textbf{Temperature} & \\textbf{top\\_k} & \\textbf{top\\_p} \\\\")
    lines.append("\\hline")
    
    for config in configurations:
        name = escape_latex(config['name'])
        sampling = "Yes" if config['do_sample'] else "No"
        temp = str(config['temperature']) if config['do_sample'] else "-"
        topk = str(config['top_k']) if config['top_k'] else "-"
        topp = str(config['top_p']) if config['top_p'] else "-"
        lines.append(f"{name} & {sampling} & {temp} & {topk} & {topp} \\\\")
        lines.append("\\hline")
    
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return '\n'.join(lines)


def format_results_table(generations, prompt, max_chars=100):
    """Generate LaTeX table for a specific prompt's results."""
    prompt_results = [g for g in generations if g['prompt'] == prompt]
    
    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\small")
    escaped_prompt = escape_latex(prompt)
    lines.append(f"\\caption{{Generation results for prompt: ``{escaped_prompt}''}}")
    lines.append("\\begin{tabular}{|l|p{10cm}|}")
    lines.append("\\hline")
    lines.append("\\textbf{Method} & \\textbf{Generated Output (truncated)} \\\\")
    lines.append("\\hline")
    
    for result in prompt_results:
        method = escape_latex(result['config_name'])
        generated = result.get('generated_only', result.get('full_output', ''))
        output_text = escape_latex(truncate_text(generated, max_chars))
        lines.append(f"{method} & {output_text} \\\\")
        lines.append("\\hline")
    
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return '\n'.join(lines)


def format_compact_table(generations, prompts, configs_to_show=None, max_chars=80):
    """Generate a more compact table showing selected configs across all prompts."""
    if configs_to_show is None:
        configs_to_show = ["Greedy", "Top-p (T=0.8, p=0.6)", "Top-p (T=1.5, p=0.9)", "Top-k (T=1.0, k=20)"]
    
    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\caption{Selected generation results across different prompts and methods}")
    lines.append("\\begin{tabular}{|l|l|p{9cm}|}")
    lines.append("\\hline")
    lines.append("\\textbf{Prompt} & \\textbf{Method} & \\textbf{Generated Output} \\\\")
    lines.append("\\hline")
    
    for prompt in prompts:
        prompt_results = [g for g in generations if g['prompt'] == prompt]
        first_in_prompt = True
        
        for result in prompt_results:
            if result['config_name'] in configs_to_show:
                prompt_display = escape_latex(truncate_text(prompt, 20)) if first_in_prompt else ""
                method = escape_latex(result['config_name'])
                generated = result.get('generated_only', '')
                output_text = escape_latex(truncate_text(generated, max_chars))
                
                lines.append(f"{prompt_display} & {method} & {output_text} \\\\")
                first_in_prompt = False
        
        lines.append("\\hline")
    
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Format generation results to LaTeX')
    parser.add_argument('--input', type=str, default='generation_results.json', help='Input JSON file')
    parser.add_argument('--output', type=str, default='generation_table.tex', help='Output LaTeX file')
    parser.add_argument('--max_chars', type=int, default=100, help='Max characters for truncation')
    parser.add_argument('--compact', action='store_true', help='Generate compact table instead of per-prompt tables')
    args = parser.parse_args()

    # Load JSON results
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    prompts = data['prompts']
    configurations = data['configurations']
    generations = data['generations']

    latex_output = []
    
    # Header
    latex_output.append("% Auto-generated LaTeX tables for Q2.8.b")
    latex_output.append("% Generated from: " + args.input)
    latex_output.append("")
    
    # Configuration summary table
    latex_output.append("% Configuration summary table")
    latex_output.append(format_config_table(configurations))
    latex_output.append("")

    if args.compact:
        # Single compact table
        latex_output.append("% Compact results table")
        latex_output.append(format_compact_table(generations, prompts, max_chars=args.max_chars))
    else:
        # Per-prompt tables
        for prompt in prompts:
            latex_output.append(f"% Results for prompt: {prompt}")
            latex_output.append(format_results_table(generations, prompt, args.max_chars))
            latex_output.append("")

    # Write output
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_output))

    print(f"LaTeX tables written to: {args.output}")


if __name__ == "__main__":
    main()
