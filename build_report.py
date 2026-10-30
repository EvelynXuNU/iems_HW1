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
    front=r'''---
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

The report follows the assignment's three technical parts. Each part includes
its model or algorithm, labeled results, quantitative validation and edge cases,
execution directions, and experiments proposed for three additional months.
Part 2 also compares gradient variance directly and reports an additional,
clearly labeled Pong tuning experiment.

**Execution overview.** Use Python 3.11 or newer. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-q2.txt
```

The shuttle can also run with the smaller requirements.txt dependency set.
The execution subsections provide training, evaluation, and validation commands.
Fresh output directories preserve the recorded experiments. CUDA commands use
exactly one assigned GPU; CPU validation is sufficient for the mathematical checks.

\newpage
\setcounter{tocdepth}{2}
\tableofcontents

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
       '--toc-depth=2','-V','geometry:margin=0.8in','-V','fontsize:10pt',
       '-V','colorlinks:true','-V','urlcolor:blue']
    subprocess.run(command,cwd=root,env=environment,check=True)
    print(output)


if __name__=='__main__':main()
