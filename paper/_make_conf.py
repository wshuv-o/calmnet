import re

s = open("calmnet_paper_journal.tex", encoding="utf-8").read()

# 1) conference document class
s = s.replace(r"\documentclass[journal]{IEEEtran}",
              r"\documentclass[conference]{IEEEtran}")

# 2) journal author block -> conference author block
conf_author = (
    r"\author{\IEEEauthorblockN{Author Name}" "\n"
    r"\IEEEauthorblockA{\textit{Department of Computer Science / Biomedical Engineering} \\" "\n"
    r"\textit{Institution Name}\\" "\n"
    r"City, Country \\" "\n"
    r"email@institution.edu}" "\n"
    r"}"
)
s = re.sub(r"\\author\{Author~Name.*?a CC0 licence\.\}\}",
           lambda m: conf_author, s, count=1, flags=re.DOTALL)

# 3) drop the journal running-head \markboth{...}{...}
s = re.sub(r"\\markboth\{.*?Self-Calibrating Framework\}\s*", "", s,
           count=1, flags=re.DOTALL)

open("calmnet_paper.tex", "w", encoding="utf-8").write(s)
ok = (r"\documentclass[conference]" in s and "IEEEauthorblockN" in s
      and r"\markboth" not in s and "---" not in s)
print("conference tex written | class+author+no-markboth+no-emdash:", ok)
