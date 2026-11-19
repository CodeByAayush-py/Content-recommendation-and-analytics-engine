"""Convert the '# %%' percent-format task scripts into .ipynb notebooks (no extra libraries needed)."""
import json, re, glob, os

def convert(path):
    cells, cur, kind = [], [], None
    def flush():
        if cur and any(l.strip() for l in cur):
            src = list(cur)
            while src and not src[-1].strip(): src.pop()
            if kind == "markdown":
                src = [re.sub(r"^# ?", "", l) for l in src]
                cells.append({"cell_type": "markdown", "metadata": {}, "source": [l if l.endswith("\n") else l + "\n" for l in src[:-1]] + [src[-1].rstrip("\n")]})
            else:
                cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                              "source": [l if l.endswith("\n") else l + "\n" for l in src[:-1]] + [src[-1].rstrip("\n")]})
    for line in open(path, encoding="utf-8"):
        m = re.match(r"# %%\s*(\[markdown\])?", line)
        if m:
            flush(); cur = []; kind = "markdown" if m.group(1) else "code"
        else:
            cur.append(line)
    flush()
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}}
    out = path.replace(".py", ".ipynb")
    json.dump(nb, open(out, "w", encoding="utf-8"), indent=1)
    print("wrote", out, f"({len(cells)} cells)")

for f in sorted(glob.glob("task*.py")):
    convert(f)
