# ACL 2027 paper draft

`main.tex` is the first English manuscript draft for the evidence-governed,
open-vocabulary knowledge-graph project. It is prepared as an anonymous ACL
review version (`\usepackage[review]{acl}`) and includes a local copy of the
ACL style interface so the draft can be compiled offline. Replace the local
style files with the current official files before submission if ACL updates
the template.

The draft deliberately distinguishes:

- results from the frozen benchmark snapshot;
- development-history evidence used to motivate the mechanisms; and
- claims that still require a controlled rerun or independent human annotation.

The current numbers should be regenerated from a frozen repository commit
before submission. The bibliography is also an initial pass; representative
LLM-based KG papers and the exact benchmark citations still need to be added.

Useful official references:

- <https://2027.aclweb.org/calls/main/>
- <https://acl-org.github.io/ACLPUB/formatting.html>
- <https://github.com/acl-org/acl-style-files>

For a local terminal build (if a TeX distribution is available):

```bash
cd paper/acl2027
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```
