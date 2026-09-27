#!/usr/bin/env python3
"""
Patch all notebooks to:
1. Add python-dotenv to packages install list
2. Add load_dotenv() before os.getenv() calls
3. Fix hardcoded Windows PDF paths to Linux relative paths
"""
import json
import re
from pathlib import Path

NOTEBOOK_ROOT = Path(__file__).parent
DATA_ROOT = NOTEBOOK_ROOT.parent / "data"


def depth_below_root(nb_path: Path) -> int:
    """How many directory levels below NOTEBOOK_ROOT is this notebook?"""
    return len(nb_path.relative_to(NOTEBOOK_ROOT).parts) - 1


def pdf_rel_path(nb_path: Path, filename: str = "climate.pdf") -> str:
    """Relative path from notebook location to data/<filename>."""
    depth = depth_below_root(nb_path)
    # Go up (depth+1) levels from notebook dir to reach rag-patterns-aws-Qdrant, then into data/
    ups = "/".join([".."] * (depth + 1))
    return f"{ups}/data/{filename}"


def env_rel_path(nb_path: Path) -> str:
    """Relative path from notebook location to qdrant_notebooks/.env."""
    depth = depth_below_root(nb_path)
    if depth == 0:
        return ".env"
    return "/".join([".."] * depth) + "/.env"


DOTENV_SNIPPET = (
    "from dotenv import load_dotenv\n"
    "load_dotenv({env_path!r})\n"
)


def cell_source(cell: dict) -> str:
    src = cell.get("source", [])
    return "".join(src) if isinstance(src, list) else src


def set_source(cell: dict, src: str):
    cell["source"] = src
    cell["outputs"] = []
    cell["execution_count"] = None


def patch_notebook(nb_path: Path) -> tuple[bool, list[str]]:
    with open(nb_path) as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    changed = False
    log = []

    # ── Pass 1: add python-dotenv to pip install cell ─────────────────────────
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        src = cell_source(cell)
        if "subprocess.check_call" not in src and "pip install" not in src:
            continue
        if "python-dotenv" in src:
            break  # already there
        # Insert "python-dotenv" before the closing ] of the packages list
        new_src = re.sub(
            r'(packages\s*=\s*\[)(.*?)(\])',
            lambda m: (
                m.group(1) + m.group(2).rstrip().rstrip(",")
                + ',\n    "python-dotenv",\n' + m.group(3)
            ),
            src,
            flags=re.DOTALL,
        )
        if new_src != src:
            set_source(cell, new_src)
            changed = True
            log.append("added python-dotenv to packages list")
        break  # only first install cell

    # ── Pass 2: add load_dotenv() in the imports cell ─────────────────────────
    env_path = env_rel_path(nb_path)
    dotenv_snippet = f'from dotenv import load_dotenv\nload_dotenv({env_path!r})\n'

    import_cell = None
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        src = cell_source(cell)
        if ("import os" in src or "import boto3" in src) and "subprocess" not in src:
            import_cell = cell
            break

    if import_cell is not None:
        src = cell_source(import_cell)
        if "load_dotenv" not in src and "dotenv" not in src:
            # Prepend dotenv loading before other imports
            new_src = dotenv_snippet + "\n" + src
            set_source(import_cell, new_src)
            changed = True
            log.append(f"added load_dotenv('{env_path}')")

    # ── Pass 3: fix Windows PDF paths ─────────────────────────────────────────
    filename_map = {
        r'climate\.pdf': "climate.pdf",
        r'PATIENT[^"\'\\]*\.pdf': "PATIENT INFORMATION SYSTEMS.pdf",
        r'medicaid[^"\'\\]*\.pdf': "medicaid.pdf",
    }
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        src = cell_source(cell)
        new_src = src

        for pattern, fname in filename_map.items():
            # Match PDF_PATH = r"...\file.pdf" or PDF_PATH = "...\file.pdf"
            # The path contains a backslash (Windows) or is an absolute path
            def make_replacer(fn):
                def replacer(m):
                    rel = pdf_rel_path(nb_path, fn)
                    return f'{m.group(1)}{rel!r}'
                return replacer

            new_src = re.sub(
                r'(PDF_PATH\s*=\s*)[rR]?"[^"]*' + pattern + '"',
                make_replacer(fname),
                new_src,
            )

        if new_src != src:
            set_source(cell, new_src)
            changed = True
            log.append("fixed hardcoded Windows PDF path")

    if changed:
        with open(nb_path, "w") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)

    return changed, log


def main():
    notebooks = sorted(NOTEBOOK_ROOT.rglob("*.ipynb"))
    patched = 0
    for nb_path in notebooks:
        rel = nb_path.relative_to(NOTEBOOK_ROOT)
        try:
            changed, log = patch_notebook(nb_path)
            if changed:
                patched += 1
                print(f"  PATCHED  {rel}")
                for msg in log:
                    print(f"           - {msg}")
            else:
                print(f"  ok       {rel}")
        except Exception as e:
            print(f"  ERROR    {rel}: {e}")
    print(f"\n{patched}/{len(notebooks)} notebooks patched.")


if __name__ == "__main__":
    main()
