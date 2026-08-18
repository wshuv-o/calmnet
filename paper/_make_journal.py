import re

src = open('calmnet_paper.tex', encoding='utf-8').read()

src = src.replace(r'\documentclass[conference]{IEEEtran}',
                  r'\documentclass[journal]{IEEEtran}')

new_author = (
    r'\author{Author~Name,~\IEEEmembership{Student~Member,~IEEE}' '\n'
    r'\thanks{Author Name is with the Department of Computer Science / Biomedical' '\n'
    r'Engineering, Institution Name, City, Country' '\n'
    r'(e-mail: email@institution.edu).}' '\n'
    r'\thanks{This work uses the NeuroRex dataset (OpenNeuro ds007788), released under' '\n'
    r'a CC0 licence.}}'
)
src2 = re.sub(r'\\author\{\\IEEEauthorblockN.*?email@institution\.edu\}\s*\}',
              lambda m: new_author, src, count=1, flags=re.DOTALL)
assert src2 != src and 'IEEEauthorblockN' not in src2, "author replace failed"

markboth = (
    r'\markboth{IEEE Transactions on Neural Systems and Rehabilitation '
    r'Engineering,~Vol.~XX,~No.~X,~2026}%' '\n'
    r'{Author \MakeLowercase{\textit{et al.}}: CALM-Net: A Calibrated, '
    r'Abstaining, and Longitudinal EEG Decoder}' '\n'
    r'\maketitle'
)
src = src2.replace(r'\maketitle', markboth, 1)

open('calmnet_paper_journal.tex', 'w', encoding='utf-8').write(src)
print("wrote calmnet_paper_journal.tex  | class:", src.splitlines()[0],
      "| author ok:", 'IEEEauthorblockN' not in src, "| markboth ok:", '\\markboth' in src)
