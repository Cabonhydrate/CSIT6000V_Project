"""Small, source-preserving PDF table index for the single-user demo.

No answer keys or paper-specific numerical constants. Tables are reconstructed
from positioned PDF words. Ambiguous structures are skipped rather than guessed.
"""
import re
from pathlib import Path
from .. import db
from ..config import settings
from ..logging_setup import get_logger

logger = get_logger('tables')

CAPTION = re.compile(r'^\s*Table\s+(\d+)\s*\.', re.I)


def _lines(page):
    words = sorted(page.get_text('words'), key=lambda w: (w[1], w[0]))
    lines = []
    for word in words:
        x0, y0, x1, y1, text = word[:5]
        if lines and abs(lines[-1]['y'] - y0) <= 3.0:
            lines[-1]['words'].append((x0, x1, text))
        else:
            lines.append({'y': y0, 'end': y1, 'words': [(x0, x1, text)]})
    for line in lines:
        line['words'].sort()
        line['text'] = ' '.join(w[2] for w in line['words'])
    return lines


def _header_columns(line):
    # Cluster words separated by a gap, recovering column boundaries from the
    # actual table header instead of assuming a fixed set of column positions.
    cells = []
    for x0, x1, text in line['words']:
        if not cells or x0 - cells[-1][-1][1] > 9.0:
            cells.append([])
        cells[-1].append((x0, x1, text))
    labels = [' '.join(w[2] for w in cell) for cell in cells]
    starts = [cell[0][0] for cell in cells]
    if not (3 <= len(labels) <= 10):
        return None
    return labels, starts


def _cell_index(x, starts):
    # Start-aligned data columns; midpoint between adjacent header starts is
    # not suitable for right-aligned numbers, so use the next start minus 8pt.
    for idx in range(len(starts)-1, -1, -1):
        if x >= starts[idx] - 9:
            return idx
    return 0


def _row_cells(line, starts):
    cells = [[] for _ in starts]
    for x0, _, text in line['words']:
        cells[_cell_index(x0, starts)].append(text)
    return [' '.join(x).strip() for x in cells]


def _table_regions(page):
    """Yield (table_number, header line, data lines) for identifiable captions."""
    lines = _lines(page)
    captions = []
    for block in page.get_text('blocks'):
        caption = CAPTION.match(block[4])
        if caption:
            captions.append((int(caption.group(1)), float(block[3]), ' '.join(block[4].split())))
    for number, caption_end, caption_text in captions:
        header = next((i for i, line in enumerate(lines)
                       if caption_end < line['y'] < caption_end + 45
                       and _header_columns(line) is not None), None)
        if header is None:
            continue
        sample = lines[header + 1:]
        rows = []
        prev = lines[header]['y']
        for line in sample:
            if line['y'] - prev > (32 if not rows else 22.5) or CAPTION.match(line['text']):
                break
            rows.append(line)
            prev = line['y']
        if len(rows) >= 2:
            yield number, caption_text, lines[header], rows


def extract_pdf_tables(path):
    """Return table rows with a real PDF page, caption id, and header mapping.

    Rejects figure data and ambiguous rows rather than treating diagram labels
    as verified table facts. Works with span/group labels (e.g., Dataset in Table 5).
    """
    import fitz
    result = []
    with fitz.open(path) as pdf:
        for page_index, page in enumerate(pdf):
            for number, caption_text, header_line, data_lines in _table_regions(page):
                parsed = _header_columns(header_line)
                if parsed is None:
                    continue
                headers, starts = parsed
                grid = [(line['y'], _row_cells(line, starts)) for line in data_lines]
                # A data row must have values in the majority of columns.
                dense = [(y, cells) for y, cells in grid if sum(bool(c) for c in cells[1:]) >= len(headers)-2]
                if len(dense) < 2:
                    continue
                # First-column labels are frequently vertical group labels
                # centered in the middle of multiple related metric rows.
                groups = [(y, cells[0]) for y, cells in grid if cells[0]]
                rows=[]
                for y, cells in dense:
                    if not cells[0] and groups:
                        cells[0] = min(groups, key=lambda t: abs(t[0]-y))[1]
                    rows.append(dict(zip(headers, cells)))
                if rows:
                    result.append({'table_number': number, 'page': page_index + 1,
                                   'headers': headers, 'rows': rows, 'caption': caption_text})
    return result


def store_tables(paper_id, tables):
    with db.connect() as con:
        con.execute('DELETE FROM paper_tables WHERE paper_id=?', (paper_id,))
        for table in tables:
            con.execute('''INSERT INTO paper_tables(paper_id, table_number, page, headers, rows_json, caption)
                           VALUES(?,?,?,?,?,?)''', (paper_id, table['table_number'], table['page'],
                           db.encode(table['headers']), db.encode(table['rows']), table.get('caption','')))
    logger.info('paper=%s tables_indexed=%d', paper_id, len(tables))


def _pdf_location(paper):
    if paper.get('source_type') != 'pdf':
        return None
    original = Path(paper.get('document_path') or '')
    candidates = [settings.data_dir / 'pdfs' / f"{paper['id']}.pdf", original]
    return next((p for p in candidates if p.is_file()), None)


def load_tables(paper_ids, number=None):
    """Auto-index existing uploads on first use; no PDF re-upload required."""
    if not paper_ids:
        return []
    output = []
    for paper_id in paper_ids:
        paper = db.one('SELECT id,source_type,document_path FROM papers WHERE id=? AND owner_id=?',
                       (paper_id, settings.owner_id))
        if not paper or paper['source_type'] != 'pdf':
            continue
        records = db.rows('SELECT * FROM paper_tables WHERE paper_id=?', (paper_id,))
        if not records:
            path = _pdf_location(paper)
            if path:
                records = extract_pdf_tables(path)
                if records:
                    store_tables(paper_id, records)
                    records = db.rows('SELECT * FROM paper_tables WHERE paper_id=?', (paper_id,))
        for row in records:
            if number is None or row['table_number'] == number:
                output.append({'paper_id': paper_id, 'table_number': row['table_number'],
                               'page': row['page'], 'headers': db.decode(row['headers'], []),
                               'rows': db.decode(row['rows_json'], []), 'caption': row.get('caption','')})
    return output
