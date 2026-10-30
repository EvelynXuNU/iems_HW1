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

The required experiments use dynamic programming for shuttle dispatching and
PyTorch REINFORCE with and without a lagged scalar baseline. Every technical
part includes its assumptions, labeled plots, quantitative validation, edge
cases, execution directions, and proposed experiments for three additional months.

**Results at a glance.** The shuttle's infinite-horizon optimum dispatches at
13 or more passengers, with expected discounted cost $J^*(0)=576.670168$.
The policy results below each use 500 frozen, stochastic evaluation episodes;
parentheses contain sample standard deviations.

| Experiment | CartPole score | Pong score |
|---|---:|---:|
| Part 1: REINFORCE, no baseline | 181.612 (97.2304) | -19.046 (2.5987) |
| Part 2: REINFORCE, lagged EMA | 438.342 (95.7679) | -19.460 (0.9952) |
| Additional CNN / EMA pilot | Not run | -18.492 (1.8722) |
| Additional single-GPU PPO | 500.000 (0.0000) | -6.626 (4.9727) |

The two required Pong runs and the CNN pilot had no wins. Fresh single-GPU
PPO training and evaluation produced 45/500 Pong wins and 500/500 CartPole games
at the cap. PPO is supplementary because it changes the algorithm and Pong
representation. These results are conditional on one training seed; Pong is
not solved and CartPole's observed zero SD is not population certainty.

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
The optional PPO study has separate dependencies in requirements-ppo.txt.

\newpage
\setcounter{tocdepth}{2}
\tableofcontents

'''
    sections=[(root/p).read_text() for p in ['q1_report.md','q2_report.md','q3_report.md']]
    supplemental=root/'results/ppo_gpu/report_section.md'
    if supplemental.exists():
        sections.append(supplemental.read_text())
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
