"""Utilities to convert space-aligned TXT table exports (like COMSOL table dumps) to CSV.

Usage:
    from txt2csv import txt_to_csv
    txt_to_csv('测试4.txt')  # writes '测试4.csv' by default

Also provides a CLI:
    python txt2csv.py input.txt -o output.csv
"""

from __future__ import annotations
import re
import csv
from pathlib import Path
from typing import List, Optional

_SPLIT_RE = re.compile(r"\s{2,}")  # split on two or more spaces


def _looks_like_data_line(line: str) -> bool:
    """Return True if all tokens in the line are parseable as numbers.

    Uses the same splitting rule as the parser (prefer split on 2+ spaces,
    fall back to any whitespace) so we correctly detect numeric lines.
    """
    toks = _SPLIT_RE.split(line.strip())
    if len(toks) == 1:
        toks = re.split(r"\s+", line.strip())
    if not toks:
        return False
    for t in toks:
        if t == '':
            return False
        try:
            float(t)
        except Exception:
            return False
    return True


def _clean_headers(headers: List[str]) -> List[str]:
    """Sanitize and make header names safe for pandas/CSV column names.

    - replace '@' with 'at'
    - remove parentheses while preserving content
    - replace non-alphanumeric characters with underscores
    - collapse multiple underscores and strip leading/trailing underscores
    - ensure uniqueness by appending suffixes when necessary
    """
    clean: List[str] = []
    used: dict[str, int] = {}

    for h in headers:
        s = h.strip()
        s = s.replace('@', '_at_')
        s = re.sub(r'[()]', '_', s)
        # replace any sequence of non-alnum with underscore
        s = re.sub(r'[^0-9A-Za-z]+', '_', s)
        s = re.sub(r'_+', '_', s).strip('_')
        if s == '':
            s = 'col'
        base = s
        if base in used:
            used[base] += 1
            s = f"{base}_{used[base]}"
        else:
            used[base] = 0
        clean.append(s)
    return clean


def _find_header_and_rows(lines: List[str], detect_header_lines: int = 3, clean_headers_flag: bool = True) -> (List[str], List[List[str]]):
    """Find header line (if present) and parse rows robustly.

    Heuristics:
    - Skip comment lines starting with '%'.
    - Inspect the first `detect_header_lines` non-comment lines: if any line
      contains a non-numeric token it is treated as the header.
    - If no header is detected, generate generic column names `col_1`.. based
      on the number of columns inferred from the first data line.
    - Data splitting: prefer splitting on 2+ spaces; fall back to any whitespace.
    - If a data row has more tokens than the header, extras are joined into
      the final column (conservative choice).
    """
    # Keep original lines to allow detecting a commented header line (COMSOL may export headers as commented lines starting with '%')
    clean_lines = [ln.rstrip() for ln in lines if ln.strip() and not ln.lstrip().startswith('%')]
    if not clean_lines:
        raise ValueError("No data found in input file")

    # 1) Try to detect a header line in comment lines (e.g., "% X  Y  Z  ...")
    header_from_comment: Optional[str] = None
    header_idx: Optional[int] = None
    for ln in lines[: max(detect_header_lines + 5, 10)]:
        if ln.lstrip().startswith('%'):
            candidate = ln.lstrip()[1:].strip()
            # skip metadata-like comments (e.g., '% Model: ...')
            if ':' in candidate:
                continue
            # if candidate is non-empty and not purely numeric, treat as header
            if candidate and not _looks_like_data_line(candidate):
                # require at least two tokens (sanity)
                toks = _SPLIT_RE.split(candidate)
                if len(toks) < 2:
                    toks = re.split(r"\s+", candidate)
                if len(toks) >= 2:
                    header_from_comment = candidate
                    break

    if header_from_comment is not None:
        header_line = header_from_comment
        headers = [h.strip() for h in _SPLIT_RE.split(header_line.strip())]
        data_lines = clean_lines
    else:
        # 2) Inspect the first few non-comment lines for a header (non-numeric)
        header_idx: Optional[int] = None
        inspect_count = min(detect_header_lines, len(clean_lines))
        for i in range(inspect_count):
            if not _looks_like_data_line(clean_lines[i]):
                header_idx = i
                break

        if header_idx is not None:
            header_line = clean_lines[header_idx]
            headers = [h.strip() for h in _SPLIT_RE.split(header_line.strip())]
            data_lines = clean_lines[header_idx + 1 :]
        else:
            # no explicit header found; infer number of columns from first data line
            first = clean_lines[0]
            parts = [p.strip() for p in _SPLIT_RE.split(first.strip())]
            if len(parts) == 1:
                parts = [p.strip() for p in re.split(r"\s+", first.strip())]
            headers = [f"col_{i+1}" for i in range(len(parts))]
            data_lines = clean_lines

    rows: List[List[str]] = []
    for ln in data_lines:
        parts = [p.strip() for p in _SPLIT_RE.split(ln.strip())]
        if len(parts) < len(headers):
            parts = [p.strip() for p in re.split(r"\s+", ln.strip())]
        if len(parts) < len(headers):
            parts += [''] * (len(headers) - len(parts))
        elif len(parts) > len(headers):
            parts = parts[: len(headers) - 1] + [" ".join(parts[len(headers) - 1 :])]
        rows.append(parts)

    if clean_headers_flag and (header_from_comment is not None or header_idx is not None):
        headers = _clean_headers(headers)

    return headers, rows


def txt_to_csv(input_path: str | Path, output_path: Optional[str | Path] = None, delimiter: str = ',', clean_headers: bool = True, detect_header_lines: int = 3) -> Path:
    """Convert a text table (like `测试4.txt`) to a CSV file.

    Parameters:
    - clean_headers: whether to sanitize header names (default: True)
    - detect_header_lines: how many initial non-comment lines to inspect for a header (default: 3)

    Returns the path to the written CSV file.
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_suffix('.csv')
    output_path = Path(output_path)

    text = input_path.read_text(encoding='utf-8')
    lines = text.splitlines()

    headers, rows = _find_header_and_rows(lines, detect_header_lines=detect_header_lines, clean_headers_flag=clean_headers)

    with output_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow(headers)
        writer.writerows(rows)

    return output_path


def convert(txt_path: str | Path, clean_headers: bool = True, detect_header_lines: int = 3, delimiter: str = ',') -> Path:
    """Convenience wrapper that converts the given TXT file to a CSV file with the same
    base name in the same directory and returns the CSV Path.

    Example:
        convert('path/to/测试4.txt')  # writes path/to/测试4.csv
    """
    return txt_to_csv(txt_path, output_path=None, delimiter=delimiter, clean_headers=clean_headers, detect_header_lines=detect_header_lines)


__all__ = ['txt_to_csv', 'convert']


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Convert space-aligned TXT table to CSV')
    parser.add_argument('input', help='Input TXT file')
    parser.add_argument('-o', '--output', help='Output CSV file (optional). Defaults to same name with .csv')
    parser.add_argument('-d', '--delimiter', default=',', help='CSV delimiter (default: ,)')
    parser.add_argument('--no-clean', dest='clean_headers', action='store_false', help='Do not clean header names (default: clean)')
    parser.add_argument('--detect-lines', type=int, default=3, help='Number of non-comment lines to inspect for a header (default: 3)')
    args = parser.parse_args()

    out = convert(args.input, clean_headers=args.clean_headers, detect_header_lines=args.detect_lines, delimiter=args.delimiter) 
    print(f'Wrote CSV to: {out}')
