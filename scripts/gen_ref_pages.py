"""Generate one API reference page per module under src/global_vae, plus a SUMMARY.md.

The literate-nav SUMMARY.md ties the pages together (spec §10 "Documentation:
mkdocs + mkdocstrings").

Run automatically by mkdocs itself (see the `gen-files` plugin entry in
mkdocs.yml) via `mkdocs build` / `mkdocs serve` -- never run this directly.
Adding a new module under src/global_vae needs no change here and no nav
edit anywhere else: this script walks the package tree fresh on every
build, mirroring the same self-registration philosophy the codebase
already uses for its own registries.
"""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

root = Path(__file__).parent.parent
src = root / "src"
package_root = src / "global_vae"
generated_count = 0


def _isDeprecatedAlias(path: Path) -> bool:
    """Whether `path` is a CamelCase shim kept at an old import path (roadmap D-9).

    Every module of the package is snake_case. The few CamelCase files are the pre-1.3 paths of
    renamed modules, kept until 2.0: each only re-exports a class that its snake_case module
    already documents, so a page for it would document the same class twice.

    Args:
        path: A Python file under `src/global_vae`.

    Returns:
        `True` when the file name has an uppercase letter.
    """
    return path.stem != path.stem.lower()


# rglob, not glob: this MUST walk every subpackage recursively (encoders/,
# decoders/, fusion/, assemblers/, latent/, losses/, data/, training/,
# models/, evaluation/, visualization/, utils/, config/, ...), not just the
# top-level __init__.py. If a future edit accidentally swaps this back to a
# non-recursive glob, the sanity check at the bottom catches it.
for path in sorted(package_root.rglob("*.py")):
    if _isDeprecatedAlias(path):
        continue
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

    nav_parts = parts[1:] if len(parts) > 1 else parts
    nav[nav_parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as reference_file:
        identifier = ".".join(parts)
        reference_file.write(f"# `{identifier}`\n\n::: {identifier}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))
    generated_count += 1

with mkdocs_gen_files.open("reference/index.md", "w") as reference_file:
    reference_file.write("# API Reference\n\n")
    reference_file.write(
        "This section documents the public Python API of Global Multimodal VAE.\n\n"
    )

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())

# Fail loudly instead of silently shipping a near-empty API reference. This
# codebase has 60+ modules under src/global_vae; a run producing only a
# handful of pages means the walk above broke (wrong package_root, glob
# instead of rglob, ...), not that the package shrank.
python_files = [
    path
    for path in package_root.rglob("*.py")
    if path.name != "__main__.py" and not _isDeprecatedAlias(path)
]
if generated_count < len(python_files):
    raise RuntimeError(
        f"gen_ref_pages.py only generated {generated_count} reference page(s) "
        f"from '{package_root}', expected 20+. Check that package_root points "
        f"at src/global_vae and that rglob (not glob) is used, so the walk is "
        f"recursive into every subpackage."
    )
