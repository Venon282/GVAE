"""Generate one API reference page per module under src/global_vae, plus a
literate-nav SUMMARY.md tying them together (spec §10 "Documentation:
mkdocs + mkdocstrings").

Run automatically by mkdocs itself (see the `gen-files` plugin entry in
mkdocs.yml) via `mkdocs build` / `mkdocs serve` — never run this directly.
Adding a new module under src/global_vae needs no change here and no nav
edit anywhere else: this script walks the package tree fresh on every
build, mirroring the same self-registration philosophy the codebase
already uses for its own registries (encoders, fusion, assemblers, ...):
a new file just needs to exist, nothing needs to be told about it by hand.
"""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

root = Path(__file__).parent.parent
src = root / "src"
package_root = src / "global_vae"


for path in sorted(package_root.rglob("*.py")):
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        if not parts:
            continue
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as reference_file:
        identifier = ".".join(parts)
        reference_file.write(f"# `{identifier}`\n\n::: {identifier}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
