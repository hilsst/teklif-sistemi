"""Command line entry point: ``teklif <command>``."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

from teklif.costing.engine import QuoteInputError, compute_quote
from teklif.costing.inputs import QuoteInput
from teklif.costing.regression import run_regression
from teklif.library import LibraryError, list_todos, load_library
from teklif.library.loader import DEFAULT_LIBRARY_DIR, Library
from teklif.library.todo_report import render_todo_markdown
from teklif.reporting.excel_templates import write_templates
from teklif.reporting.quote_report import fmt, summary_rows, write_excel, write_pdf
from teklif.storage.db import Store
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
    qt = sub.add_parser("quote", help="Compute a quote from a YAML input file")
    qt.add_argument("input", type=Path)
    qt.add_argument("--xlsx", type=Path, help="Write the Excel breakdown")
    qt.add_argument("--pdf", type=Path, help="Write the PDF quote (needs the 'pdf' extra)")
    qt.add_argument("--db", type=Path, help="Save the quote to this SQLite file")
    rg = sub.add_parser("regress", help="Compare the reference set with actual cost")
    rg.add_argument("file", type=Path)
    sub.add_parser("app", help="Start the Streamlit UI (needs the 'app' extra)")
    args = parser.parse_args(argv)

    if args.cmd == "app":
        import subprocess

        app = Path(__file__).resolve().parent / "app" / "streamlit_app.py"
        env = {**os.environ, "TEKLIF_LIBRARY": str(args.library)}
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)], env=env)

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
    elif args.cmd == "quote":
        return _quote(args, lib)
    elif args.cmd == "regress":
        return _regress(args.file, lib)
    return 0


def _quote(args: argparse.Namespace, lib: Library) -> int:
    data = yaml.safe_load(args.input.read_text(encoding="utf-8"))
    try:
        result = compute_quote(QuoteInput.model_validate(data), lib)
    except (QuoteInputError, ValueError) as exc:
        print(f"Girdi hatası: {exc}", file=sys.stderr)
        return 1
    for row in summary_rows(result):
        mark = "  [manual_review]" if row.review else ""
        print(f"{row.label:32s} {fmt(row.value):>14s} {row.unit}{mark}")
    if result.manual_review:
        print(f"\nManuel inceleme: {len(result.review_items)} bayrak")
        for key, flag in result.review_items[:15]:
            print(f"  {key}: {flag.code} {flag.path or flag.reason}")
        if len(result.review_items) > 15:
            print("  ... (tamamı Excel kırılımında)")
    if args.xlsx:
        print(f"Wrote {write_excel(result, args.xlsx)}")
    if args.pdf:
        print(f"Wrote {write_pdf(result, args.pdf)}")
    if args.db:
        saved = Store(args.db).save_quote(result, lib)
        print(f"Saved quote {saved.quote_no} rev {saved.revision} (id {saved.id})")
    return 0


def _regress(path: Path, lib: Library) -> int:
    report = run_regression(path, lib)
    for issue in report.import_issues:
        print(f"satır {issue.row}, {issue.column}: {issue.message}")
    for r in report.rows:
        dev = "—" if r.deviation is None else f"{r.deviation:+.1%}"
        print(
            f"{r.quote_no:14s} {r.part_no:14s} {r.status:17s} hesap {fmt(r.computed):>12s} "
            f"fiili {fmt(r.actual):>12s} sapma {dev:>7s} {r.reason}"
        )
    n, ok = len(report.compared), len(report.within)
    print(f"\nKarşılaştırılan {n} iş, ±{report.tolerance:.0%} içinde {ok}.")
    return 0 if n and ok == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
