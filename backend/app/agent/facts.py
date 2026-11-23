"""Evidence-first table facts: select one aligned row before doing arithmetic.

This is not a document-specific answer database. All values come from parsed
PDF cells; unclear row/column matches return None so the normal Agent can try.
"""
import re
from decimal import Decimal, InvalidOperation
from .. import db
from ..knowledge import retrieval
from ..knowledge.tables import load_tables


def _normalize(text):
    return ''.join(re.findall(r'[a-z0-9]+', str(text).casefold()))


def _mentioned(question, value):
    n = _normalize(value)
    return bool(len(n) >= 4 and n in _normalize(question))


def select_fact_row(question, paper_ids):
    match = re.search(r'\btable\s*(\d+)\b|表\s*(\d+)', question, re.I)
    if not match:
        return None
    number = int(match.group(1) or match.group(2))
    candidates = load_tables(paper_ids, number)
    scored = []
    for table in candidates:
        # Dataset and Metric/Characteristic usually occupy the first two columns.
        # Use both when the question names both; never take values merely because
        # they occur on the same PDF page.
        for row in table['rows']:
            labels = [str(row.get(h, '')) for h in table['headers'][:2]]
            matches = [_mentioned(question, label) for label in labels]
            if not any(matches):
                continue
            # 'Context Precision (w/o GT)' must not match a request for the
            # distinct plain 'Context Precision' row.
            if any('w/o' in label.casefold() and 'w/o' not in question.casefold()
                   and _mentioned(question, label.replace('(w/o GT)', '').strip())
                   for label in labels):
                continue
            # Prefer rows where all named descriptors are present; if the
            # user specifies one key, other unmatched rows stay possible.
            score = sum(100 + len(_normalize(label)) for label, yes in zip(labels, matches) if yes)
            scored.append((score, table, row))
    if not scored:
        return None
    scored.sort(key=lambda x: -x[0])
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return None  # Multiple equally plausible rows: do not invent alignment.
    _, table, row = scored[0]
    return {'table': table, 'row': row}


def _descriptors(fact):
    headers = fact['table']['headers']
    # Determine descriptors from source cell types, rather than assuming every
    # paper table begins with exactly two text columns.
    count = 0
    for header in headers:
        if _decimal(fact['row'].get(header, '')) is not None:
            break
        count += 1
    return max(1, count)


def requested_columns(question, fact):
    """Map user-facing strategy names to the table's own column labels."""
    q = question.casefold()
    out = []
    for header in fact['table']['headers'][_descriptors(fact):]:
        h = header.casefold()
        named = _mentioned(question, header)
        if h == 'r':
            named = 'random' in q or re.search(r'(?<![a-z])r(?![a-z])', q) is not None
        elif h == 'cc':
            named = ('cc-guided' in q or 'coverage-guided' in q or
                     'cc引导' in q or 'cc-guided' in q.replace(' ', ''))
        elif h == 'ob':
            named = 'overlap' in q or 'ob' in q.split()
        elif h == 'coverage-guided':
            named = 'cc-guided' in q or 'coverage-guided' in q
        elif h == 'overlap-biased':
            named = 'overlap' in q
        # Never treat a request mentioning CC-guided as the unrelated
        # 'Speedup (R)'/'Speedup (OB)' column.
        if h.startswith('speedup'):
            named = ('speedup' in q and _mentioned(question, header))
        if named and header not in out:
            out.append(header)
    return out


def _decimal(text):
    try:
        cleaned = str(text).replace(',', '').strip().rstrip('×x%')
        return Decimal(cleaned)
    except (InvalidOperation, ValueError):
        return None


def _name(header):
    aliases = {'CC': 'CC-guided', 'R': 'Random', 'OB': 'Overlap-biased'}
    return aliases.get(header, header)


def table_row_quote(fact):
    table = fact['table']; row = fact['row']
    return (f"PDF 表格单元格转录：Table {table['table_number']}，第{table['page']}页\n"
            + ' | '.join(f'{h}: {row.get(h, "")}' for h in table['headers']))


def fact_context(fact):
    return (table_row_quote(fact)
            + '\n生成回答时，请按此表格行核对数据集、指标和数值列。')


def direct_table_answer(question, fact):
    """Answer pure numeric lookups/computations without making the LLM pick cells.

    Explanatory questions remain with the Agent, supplied with verified cells.
    """
    q = question.casefold()
    # E.g. "why does this exception occur?" needs paragraph evidence too.
    if any(w in q for w in ('为什么', '原因', '这一例外', '为何', 'why ', 'how did', '怎么解释')):
        return None
    cols = requested_columns(question, fact)
    if len(cols) < 2:
        return None
    row = fact['row']
    values = {col: _decimal(row.get(col, '')) for col in cols}
    if any(value is None for value in values.values()):
        return None
    # Answer only requests that explicitly ask for numeric results.
    if not any(w in q for w in ('多少', '几次', '计算', '加速比', '百分比', '百分点',
                                'how many', 'compute', 'calculate', 'speedup', 'percentage')):
        return None
    table = fact['table']
    keys = table['headers'][:_descriptors(fact)]
    subject = '、'.join(str(row[k]) for k in keys if row.get(k))
    shown = '，'.join(f'{_name(c)} = {row[c]}' for c in cols)
    text = f"根据 Table {table['table_number']}（PDF 第{table['page']}页），{subject}：{shown}。"
    guided = next((c for c in cols if c.casefold() in ('cc','coverage-guided')), None)
    random = next((c for c in cols if c.casefold() in ('r','random')), None)
    if guided and random and values[guided] and values[random]:
        a, b = values[guided], values[random]
        if '加速比' in q or 'speedup' in q or '倍数' in q or '多少倍' in q:
            # Only treat "fewer" as faster when the table describes cost,
            # elapsed time or test executions rather than arbitrary scores.
            caption = table.get('caption','').casefold()
            if (not all((a >= 1,b >= 1)) or
                    not re.search(r'number of tests|required|time|runtime|latency|executions|耗时|测试次数',caption)):
                return None
            ratio = b / a
            text += f' 相对 Random 的加速比 = {row[random]}/{row[guided]} ≈ {ratio:.2f} 倍。'
        if any(w in q for w in ('相对提升', '提升百分比', 'percent', 'percentage', '百分点')):
            change = (a - b) / b * 100
            text += f' 相对提升 = ({row[guided]}−{row[random]})/{row[random]} × 100% ≈ {change:.2f}%。'
            if '百分点' in q:
                pp=(a-b)*100 if 0<=a<=1 and 0<=b<=1 else a-b
                multiplier=' × 100' if 0<=a<=1 and 0<=b<=1 else ''
                text += f' 绝对提升 = ({row[guided]}−{row[random]}){multiplier} = {pp:.1f} 个百分点。'
    if '解释这一结果' in q or '说明了什么' in q:
        text += ' 在同一目标覆盖率下，所需测试查询次数越少，表示覆盖获取效率越高；该表本身不证明答案生成准确率提升。'
    return text


def verified_table_evidence(fact, available):
    """Get a real chunk citation for the table page, not a synthetic ref."""
    table = fact['table']
    hits = [ev for ev in available if ev.get('paper_id') == table['paper_id']
            and table['page'] in ev.get('page_numbers', [])]
    for chunk in db.rows('SELECT * FROM chunks WHERE paper_id=? ORDER BY seq', (table['paper_id'],)):
        if table['page'] in db.decode(chunk['pages'], []):
            if any(ev['evidence_id']==chunk['id'] for ev in hits):
                continue
            hits.append({'evidence_id': chunk['id'], 'paper_id':chunk['paper_id'],
                         'chunk_id':chunk['id'], 'section':chunk['section'],
                         'page_numbers':db.decode(chunk['pages'],[]),
                         'paragraph_refs':db.decode(chunk['paragraph_refs'],[]),
                         'content':chunk['content']})
    if not hits:
        return None
    key=f"table {table['table_number']}"
    labels=' '.join(str(table_row.get(h,'')) for h in table['headers'][:2]
                    for table_row in table['rows'][:1]).casefold()
    return max(hits,key=lambda ev:(4*int(key in ev['content'].casefold())
                                  +sum(2 for word in labels.split() if len(word)>3 and word in ev['content'].casefold())))


def numeric_row_conflict(answer, fact):
    """Reject decimal measurements that belong to another row of the table.

    Computed values (percentages, ratios) are not generally source measurements,
    so only inspect 0.xxx measurements (typical for normalized metrics). This is
    a narrow high-precision guard, not a universal fact-checker.
    """
    source = {value for value in (_decimal(value) for value in fact['row'].values())
              if value is not None}
    # An explicitly shown subtraction result can be an additional correct number.
    derived = {abs(a-b) for a in source for b in source if a <= 1 and b <= 1}
    for match in re.finditer(r'(?<![\d.])0\.\d{2,5}(?!\d)', answer):
        if Decimal(match.group()) not in source | derived:
            return f'数值 {match.group()} 不属于请求的 Table {fact["table"]["table_number"]} 对应数据行'
    for column in requested_columns(answer, fact):
        expected = _decimal(fact['row'][column])
        aliases = [_name(column)]
        if column.casefold() == 'cc':
            aliases = ['CC-guided', 'Coverage-guided', 'CC']
        elif column.casefold() == 'r':
            aliases = ['Random', 'R']
        elif column != _name(column):
            aliases.append(column)
        for alias in aliases:
            if len(alias) < 2:
                continue
            pattern = (r'(?i)(?<![A-Za-z])'+re.escape(alias)+
                       r'(?![A-Za-z])\s*(?:的\s*(?:APFD|指标|数值|得分))?\s*'
                       r'(?:=|为|是|:|：)\s*([0-9][0-9,.]*)')
            for match in re.finditer(pattern, answer):
                claimed = _decimal(match.group(1))
                if claimed is not None and claimed != expected:
                    return f'{alias} 的数值为 {match.group(1)}，但 Table 中对应列为 {fact["row"][column]}'
    return ''



def claim_table_quote(table):
    """Source-only citation excerpt; never append an LLM instruction as PDF text."""
    header = f"PDF 表格单元格转录：Table {table['table_number']}，第{table['page']}页"
    rows = [' | '.join(f'{h}: {row.get(h, "")}' for h in table['headers'])
            for row in table['rows']]
    return header + '\n' + '\n'.join(rows)


def numeric_claim_conflict(answer, sources):
    """Validate numeric claims against *all* relevant tables, not one arbitrary row.

    For multi-table questions, first check that normalized measurements occur in
    a selected source row. Only apply strict column checks to a clause that
    actually identifies a unique metric row. Computed values and ambiguous
    prose are not treated as source-table cells.
    """
    if not isinstance(answer, str) or not sources:
        return ''
    def ratio_result(match, text):
        # A number after an explicit division is a derived result, not a
        # measurement quoted from a source row. Verify the arithmetic instead.
        prefix=text[max(0, match.start()-60):match.start()]
        formula=re.search(r'(\d[\d,]*)\s*/\s*(\d[\d,]*)\s*(?:=|≈|~)\s*$',prefix)
        if not formula:
            return None
        a,b=Decimal(formula.group(1).replace(',','')),Decimal(formula.group(2).replace(',',''))
        if b==0 or abs(a/b-Decimal(match.group()))>Decimal('0.002'):
            return f'计算式 {formula.group(1)}/{formula.group(2)} 与结果 {match.group()} 不一致'
        return ''
    allowed = set()
    for table in sources:
        for row in table['rows']:
            nums = [_decimal(value) for value in row.values()]
            nums = [n for n in nums if n is not None]
            allowed.update(nums)
            # Allow a simple difference of two normalized measurements.
            allowed.update(abs(a-b) for a in nums for b in nums
                           if 0 <= a <= 1 and 0 <= b <= 1)
    for match in re.finditer(r'(?<![\d.])0\.\d{2,5}(?!\d)', answer):
        arithmetic=ratio_result(match,answer)
        if arithmetic is not None:
            if arithmetic:
                return arithmetic
            continue
        if Decimal(match.group()) not in allowed:
            return f'数值 {match.group()} 未出现在所选表格的相关行中'

    # Split only at sentence/list boundaries, never at decimal points.
    clauses = re.split(r'[。；;！？\n]+', answer)
    for clause in clauses:
        if not clause.strip():
            continue
        table_refs = {int(value) for group in re.findall(
            r'\btable\s*(\d+)\b|表\s*(\d+)', clause, re.I) for value in group if value}
        if len(table_refs) == 1:
            matching_tables = [t for t in sources if t['table_number'] in table_refs]
            if matching_tables:
                allowed_in_table = {_decimal(value) for t in matching_tables
                                    for row in t['rows'] for value in row.values()}
                # Only check measurement-like decimals; a derived ratio is not a
                # source cell and must not be mistaken for a new table number.
                for m in re.finditer(r'(?<![\d.])0\.\d{2,5}(?!\d)', clause):
                    if ratio_result(m,clause) is not None:
                        continue
                    if Decimal(m.group()) not in allowed_in_table and Decimal(m.group()) in allowed:
                        return f'数值 {m.group()} 与本句引用的 Table {next(iter(table_refs))} 不符'

        # An explicitly mentioned metric identifies a row more precisely than
        # the dataset label. This catches Random/CC value swaps in that row.
        targets=[]
        for table in sources:
            headers=table['headers']
            if len(headers)<2 or not any(_decimal(r.get(headers[1])) is None for r in table['rows']):
                continue
            for row in table['rows']:
                metric=str(row.get(headers[1],''))
                if _mentioned(clause, metric):
                    targets.append({'table':table,'row':row})
        if len(targets) == 1:
            issue=numeric_row_conflict(clause,targets[0])
            if issue:
                return issue
    return ''


def claim_table_sources(question, paper_ids, limit=2):
    """Collect relevant factual rows for disputed multi-part claims.

    Does not decide linguistic truth on its own. Numeric comparisons are exact;
    the LLM must still explain what the study's conclusions do and don't say.
    """
    q = question.casefold()
    if not re.search(r'table\s*\d+|表\s*\d+|apfd|平均|加速|速度|倍|\d+\.\d+\s*[×x]', q):
        return []
    numbers = set(re.findall(r'\d+(?:\.\d+)?', q))
    named = re.findall(r'[a-z][a-z0-9-]{3,}', q)
    stop = {'table', 'paper', 'study', 'claims', 'these', 'results', 'dataset'}
    terms = [t for t in named if t not in stop]
    ranked=[]
    for table in load_tables(paper_ids):
        description=(table.get('caption') or '').casefold()
        headers=' '.join(table['headers']).casefold()
        cells=' '.join(' '.join(str(v) for v in row.values()) for row in table['rows']).casefold()
        # Numeric values and named caption terms are stronger than a generic
        # word shared across all tables.
        score=sum(6 for n in numbers if re.search(r'(?<![\d.])'+re.escape(n)+r'(?!\d)',cells))
        score+=sum(4 for t in terms if t in description)
        score+=sum(2 for t in terms if t in headers)
        score+=sum(1 for t in terms if t in cells)
        if score:
            ranked.append((score,table))
    ranked.sort(key=lambda item:-item[0])
    chosen=[]
    for _,table in ranked[:limit]:
        rows=table['rows']
        descriptors=table['headers'][:2]
        explicit=[row for row in rows if any(_mentioned(question,row.get(h,'')) for h in descriptors)]
        if explicit:
            # Preserve an Average row if the question contrasts an aggregate
            # with an individual dataset; it is a verified source cell, not an
            # inferred aggregate computed from rounded data.
            if ('平均' in q or 'average' in q or '倍' in q or '×' in q or
                any(re.search(r'(?<![\d.])'+re.escape(n)+r'(?!\d)',str(v))
                    for n in numbers for r in rows if _normalize(r.get(descriptors[0],''))=='average'
                    for v in r.values())):
                aggregate=[r for r in rows if any(_normalize(r.get(h,''))=='average' for h in descriptors)]
                explicit=list({id(r):r for r in (explicit+aggregate)}.values())
            rows=explicit[:7]
        else:
            rows=rows[:6]
        chosen.append(dict(table,rows=rows))
    return chosen


def claim_table_context(question, sources):
    """Formatted source rows plus a machine-checked counterexample when possible."""
    excerpts=[]
    for table in sources:
        rows=table['rows']
        lines=[f"Table {table['table_number']} / PDF page {table['page']} / {table.get('caption','')}"]
        lines.extend(' | '.join(f'{h}: {row.get(h, "")}' for h in table['headers']) for row in rows)
        columns=table['headers']
        # If both R and CC are numerical measures, mechanically check the
        # universal comparison in the retrieved rows. Never assert that these
        # measures should always be maximized for unrelated tables.
        if ('所有' in question or '全部' in question or 'every ' in question.lower()) and 'apfd' in question.casefold():
            if 'R' in columns and 'CC' in columns:
                exceptions=[]
                for row in rows:
                    left,right=_decimal(row.get('CC','')),_decimal(row.get('R',''))
                    if left is not None and right is not None and left<=right:
                        labels=' / '.join(row.get(col,'') for col in columns[:2] if row.get(col))
                        exceptions.append(f'{labels}: CC={left} <= R={right}')
                if exceptions:
                    lines.append('机器核对的反例（仅针对所列行）：'+'；'.join(exceptions))
        excerpts.append('\n'.join(lines))
    return '\n\n'.join(excerpts)
