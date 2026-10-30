"""Build the report PDF from the three Markdown sections; no training is run."""
from pathlib import Path
import os
import shutil
import subprocess


def main():
    root=Path(__file__).resolve().parent
    temporary=root/'tmp/report'
    temporary.mkdir(parents=True,exist_ok=True)
    output=root/'output/pdf/assignment1_report.pdf'
    output.parent.mkdir(parents=True,exist_ok=True)
    front='''---
title: "Assignment 1: Reinforcement Learning"
author: "Jianong Xu (GitHub: EvelynXuNU)"
date: "October 29, 2026"
---

Repository: **iems_HW1**

Source and experiment evidence: <https://github.com/EvelynXuNU/iems_HW1>

This report covers the shuttle MDP and both policy-gradient experiments.
All required reference-run figures and 500-game policy evaluations are included.
The Pong reference configurations lost every evaluation game; they are documented
as limited-learning results. Proposed improvements are distinguished from measured
experiments. Source, checkpoints, episode data, and validation evidence accompany
the report in the repository.

'''
    sections=[(root/p).read_text() for p in ['q1_report.md','q2_report.md','q3_report.md']]
    # Normalize math delimiters for older Pandoc readers.
    sections=[s.replace('\\[','$$').replace('\\]','$$').replace('\\(','$').replace('\\)','$') for s in sections]
    markdown=temporary/'assignment1_report.md'
    markdown.write_text(front+'\n\n\\newpage\n\n'+'\n\n\\newpage\n\n'.join(sections))
    environment=os.environ.copy()
    environment['TEXMFVAR']=str(temporary/'texmf-var')
    environment['TEXMFCONFIG']=str(temporary/'texmf-config')
    pandoc=shutil.which('pandoc')
    xelatex=shutil.which('xelatex')
    if not pandoc or not xelatex:raise RuntimeError('Install Pandoc and XeLaTeX to build the PDF')
    command=[pandoc,str(markdown),'-o',str(output),
       '--from=markdown+tex_math_single_backslash-implicit_figures','--standalone',
       '--pdf-engine='+xelatex,'--resource-path='+str(root),
       '--include-in-header='+str(root/'report/header.tex'),
       '--lua-filter='+str(root/'report/keep_figures.lua'),
       '--toc','--toc-depth=2','-V','geometry:margin=0.8in','-V','fontsize:10pt',
       '-V','colorlinks:true','-V','urlcolor:blue']
    subprocess.run(command,cwd=root,env=environment,check=True)
    print(output)


if __name__=='__main__':main()
