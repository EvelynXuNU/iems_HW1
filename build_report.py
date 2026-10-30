"""Build the concise Canvas report; detailed notes stay on GitHub."""
from pathlib import Path
import os
import shutil
import subprocess


def main():
    root=Path(__file__).resolve().parent
    temporary=root/'tmp/report';temporary.mkdir(parents=True,exist_ok=True)
    output=root/'output/pdf/assignment1_report.pdf';output.parent.mkdir(parents=True,exist_ok=True)
    text=(root/'report/submission.md').read_text()
    text=text.replace('\\[','$$').replace('\\]','$$').replace('\\(','$').replace('\\)','$')
    markdown=temporary/'assignment1_report.md';markdown.write_text(text)
    environment=os.environ.copy()
    environment['TEXMFVAR']=str(temporary/'texmf-var')
    environment['TEXMFCONFIG']=str(temporary/'texmf-config')
    pandoc=shutil.which('pandoc');xelatex=shutil.which('xelatex')
    if not pandoc or not xelatex:raise RuntimeError('Install Pandoc and XeLaTeX')
    subprocess.run([pandoc,str(markdown),'-o',str(output),
        '--from=markdown+tex_math_single_backslash-implicit_figures','--standalone',
        '--pdf-engine='+xelatex,'--resource-path='+str(root),
        '--include-in-header='+str(root/'report/header.tex'),
        '--lua-filter='+str(root/'report/keep_figures.lua'),
        '-V','geometry:margin=0.75in','-V','fontsize:10pt',
        '-V','colorlinks:true','-V','urlcolor:blue'],cwd=root,env=environment,check=True)
    print(output)


if __name__=='__main__':main()
