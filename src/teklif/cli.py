"""Command line entry point: ``teklif <command>``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from teklif.library import LibraryError, list_todos, load_library
from teklif.library.loader import DEFAULT_LIBRARY_DIR
from teklif.library.todo_report import render_todo_markdown
from teklif.reporting.excel_templates import write_templates
from teklif.storage.history_import import read_workbook
from teklif.storage.history_schema import WORKBOOKS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="teklif")
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY_DIR)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate", help="Validate the YAML library")
    todo = sub.add_parser("todo", help="List library values still waiting for company data")
    todo.add_argument("--out", type=Path, help="Write Markdown to this file")
    tpl = sub.add_parser("templates", help="Write empty upload workbooks")
    tpl.add_argument("--out", type=Path, default=Path("templates"))
    chk = sub.add_parser("check-upload", help="Validate a filled-in upload workbook")
    chk.add_argument("kind", choices=sorted(WORKBOOKS))
    chk.add_argument("file", type=Path)
    args = parser.parse_args(argv)

    try:
        lib = load_library(args.library)
    except LibraryError as exc:
        print(exc, file=sys.stderr)
        return 1

    if args.cmd == "validate":
        print(f"OK: {args.library} ({len(list_todos(lib))} TODO values)")
    elif args.cmd == "todo":
        md = render_todo_markdown(lib)
        if args.out:
            args.out.write_text(md, encoding="utf-8")
            print(f"Wrote {args.out}")
        else:
            print(md)
    elif args.cmd == "templates":
        for path in write_templates(lib, args.out):
            print(f"Wrote {path}")
    elif args.cmd == "check-upload":
        result = read_workbook(args.file, WORKBOOKS[args.kind], lib)
        for issue in result.issues:
            print(f"satır {issue.row}, {issue.column}: {issue.message}")
        print(f"{len(result.rows)} satır okundu, {len(result.issues)} sorun.")
        return 0 if result.ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
