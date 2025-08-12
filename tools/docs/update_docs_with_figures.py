#!/usr/bin/env python
"""
Documentation Figure Updater for JAX C-GEM

This script automatically updates the documentation with the latest generated figures.
It can be run after generating new publication figures to ensure the documentation
always shows the most recent results.

Usage:
    python tools/docs/update_docs_with_figures.py

Author: GitHub Copilot
"""

import os
import sys
import shutil
from pathlib import Path
import datetime
import re
import argparse

def check_publication_figures():
    """Check that the publication figures directory exists."""
    figures_dir = Path('OUT/Publication/figures')
    if not figures_dir.exists():
        print(f"❌ Publication figures directory not found: {figures_dir}")
        return False
    
    # Check for expected figures
    hydro_figure = next(figures_dir.glob("*hydrodynamics_transport*.png"), None)
    wq_figure = next(figures_dir.glob("*water_quality*.png"), None)
    
    if not hydro_figure or not wq_figure:
        print(f"❌ Expected figures not found in {figures_dir}")
        return False
    
    print(f"✅ Found publication figures in {figures_dir}")
    return True

def update_readme(publication_dir, _):
    """Update the README.md file with the latest figure information."""
    readme_path = Path('README.md')
    if not readme_path.exists():
        print(f"❌ README.md not found")
        return False
        
    # Read the README content
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Check for publication figures README to get descriptions
    pub_readme = Path(publication_dir) / "README.md"
    figure_descriptions = {}
    if pub_readme.exists():
        with open(pub_readme, 'r', encoding='utf-8') as f:
            pub_content = f.read()
            # Extract descriptions
            for fig_match in re.finditer(r'### (Figure \d+:.+?)\n', pub_content):
                figure_descriptions[f"Figure {fig_match.group(1).split(':')[0].split()[1]}"] = fig_match.group(1)
    
    # Update the README with latest generation date
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    content = re.sub(r'(## Publication-Quality Visualization.*?\n\n)', 
                    f'\\1**Figures generated on {current_date}**\n\n', 
                    content, flags=re.DOTALL)
    
    # Update figure descriptions if available
    for fig_num, description in figure_descriptions.items():
        pattern = fr'\*\*(Figure {fig_num[6:]}:).*?\*\*'
        if re.search(pattern, content):
            content = re.sub(pattern, f'**{description}**', content)
    
    # Update figure paths to use direct OUT paths
    content = re.sub(r'!\[Publication Quality [^]]+\]\(\.\./[^)]+\)', 
                    r'![Publication Quality Hydrodynamics](OUT/Publication/figures/figure_1_hydrodynamics_transport_comprehensive.png)', 
                    content)
    
    # Write updated content
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Updated README.md with latest figure information")
    return True

def update_docs_pages(publication_dir, _):
    """Update the docs pages with the latest figure information."""
    # --- NEW: Copy figures to docs/figures ---
    figures_src = Path('OUT/Publication/figures')
    figures_dst = Path('docs/figures')
    figures_dst.mkdir(exist_ok=True)
    for fig in figures_src.glob('*.png'):
        shutil.copy2(fig, figures_dst / fig.name)

    # Update quick-start.md
    quick_start_path = Path('docs/quick-start.md')
    if quick_start_path.exists():
        with open(quick_start_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # Update the figure path to use docs/figures
        content = re.sub(r'!\[Publication Quality Figure - Hydrodynamics\]\((.*?)\)',
                        r'![Publication Quality Figure - Hydrodynamics](figures/figure_1_hydrodynamics_transport_comprehensive.png)',
                        content)
        # Update the figure generation date
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        content = re.sub(r'\*Figures last generated on \d{4}-\d{2}-\d{2}\*\n\n', '', content)
        content = re.sub(r'(### Publication-Quality Figures.*?\n\n)',
                        f'\1*Figures last generated on {current_date}*\n\n',
                        content, flags=re.DOTALL)
        with open(quick_start_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Updated quick-start.md with latest figure information")

    # Update results.md
    results_path = Path('docs/results.md')
    if results_path.exists():
        with open(results_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # Update the figure paths to use docs/figures
        content = re.sub(r'!\[Publication Quality Hydrodynamics\]\((.*?)\)',
                        r'![Publication Quality Hydrodynamics](figures/figure_1_hydrodynamics_transport_comprehensive.png)',
                        content)
        content = re.sub(r'!\[Publication Quality Water Quality\]\((.*?)\)',
                        r'![Publication Quality Water Quality](figures/figure_2_water_quality_comprehensive.png)',
                        content)
        # Get figure captions from publication directory
        captions_file = Path(publication_dir) / "figure_captions.txt"
        if captions_file.exists():
            with open(captions_file, 'r', encoding='utf-8') as f:
                captions = f.read()
            # Extract captions for Figure 1 and Figure 2
            fig1_caption_match = re.search(r'Figure 1:.*?\n(.*?)(?=\n\n)', captions, re.DOTALL)
            fig2_caption_match = re.search(r'Figure 2:.*?\n(.*?)(?=\n\n|$)', captions, re.DOTALL)
            if fig1_caption_match:
                fig1_caption = fig1_caption_match.group(1).strip()
                content = re.sub(r'(\*\*Figure 1:\*\*)(.*?)(?=\n\n)',
                                f'\1 {fig1_caption}',
                                content, flags=re.DOTALL)
            if fig2_caption_match:
                fig2_caption = fig2_caption_match.group(1).strip()
                content = re.sub(r'(\*\*Figure 2:\*\*)(.*?)(?=\n\n)',
                                f'\1 {fig2_caption}',
                                content, flags=re.DOTALL)
        # Update the figure generation date
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        content = re.sub(r'\*Figures last generated on \d{4}-\d{2}-\d{2}\*\n\n', '', content)
        content = re.sub(r'(# Understanding Results.*?\n\n)',
                        f'\1*Figures last generated on {current_date}*\n\n',
                        content, flags=re.DOTALL)
        with open(results_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Updated results.md with latest figure information and captions")

    # Update index.md
    index_path = Path('docs/index.md')
    if index_path.exists():
        with open(index_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # Update the figure path to use docs/figures
        content = re.sub(r'!\[Publication Quality Hydrodynamics\]\((.*?)\)',
                        r'![Publication Quality Hydrodynamics](figures/figure_1_hydrodynamics_transport_comprehensive.png)',
                        content)
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Updated index.md with latest figure information")

    return True

# No longer needed - interactive visualization is shown directly in a window
# We've removed the static screenshot reference

def update_documentation(publication_dir="OUT/Publication"):
    """Main function to update documentation with latest figures."""
    print("\n" + "="*70)
    print("📄 JAX C-GEM DOCUMENTATION FIGURE UPDATER")
    print("="*70)
    
    # Check that publication figures exist
    if not check_publication_figures():
        print("❌ Cannot update documentation: publication figures not found")
        return False
    
    # Update README.md
    update_readme(publication_dir, None)
    
    # Update docs pages
    update_docs_pages(publication_dir, None)
    
    print("\n✅ Documentation updated with latest figures!")
    print(f"📊 Publication figures referenced from: {publication_dir}")
    print("="*70)
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Update JAX C-GEM documentation with latest figures')
    parser.add_argument('--publication-dir', default='OUT/Publication',
                       help='Directory containing publication figures')
    
    args = parser.parse_args()
    update_documentation(args.publication_dir)
