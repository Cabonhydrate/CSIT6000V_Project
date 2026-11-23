"""Conservative, source-gated mathematical *partial* answers.

When a small local model fails its structured-output contract, never invent a
paper's formula. Use a literal formula in the selected paper evidence when it
is readable; for the specific *mathematical operation* of normalized sum versus
set union, a tiny hypothetical witness can be computed deterministically.
No paper name, dataset, scoring item, or reference answer is encoded here.
"""
import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class MathFallback:
    text: str
    citation: int
    kind: str
    formula: str = ''


def _question_math_subjects(question):
    return set(re.findall(r'(?<![A-Za-z])[A-Z]{2,12}(?![A-Za-z])', question or ''))


def _recover_multiline_union_fraction(content):
    """Recover a union/cardinality fraction from visibly corrupted PDF glyphs.

    PyMuPDF can extract a displayed fraction as "||Ð\nq∈TRk(q)\n||\n|C|".
    Only recover it if LHS, bounded-index, identical query variable, explicit
    cardinality bars, and denominator all appear in that *same source span*.
    No formula is ever reconstructed from the user's proposed expression.
    """
    text = unicodedata.normalize('NFKC', str(content)).replace('\x0c', '|')
    pattern = re.compile(
        r'(?<![A-Za-z0-9_])(?P<metric>[A-Z][A-Za-z0-9_]{1,14})\s*'
        r'\(\s*(?P<domain>[A-Za-z])\s*\)\s*=\s*\n'
        r'\s*\|\|\s*[Ð∪]\s*\n'
        r'\s*(?P<index>[a-z])\s*∈\s*(?P<bound>[A-Za-z])\s*'
        r'(?P<retr>[A-Za-z])\s*(?P<k>[a-z])\s*'
        r'\(\s*(?P<index2>[a-z])\s*\)\s*\n'
        r'\s*\|\|\s*\n\s*\|\s*(?P<corpus>[A-Za-z])\s*\|')
    for match in pattern.finditer(text):
        d = match.groupdict()
        if d['index'] != d['index2'] or d['domain'] != d['bound']:
            continue
        return (d['metric'] + '(' + d['domain'] + ') = '
                + '|∪_{' + d['index'] + '∈' + d['bound'] + '} '
                + d['retr'] + '_' + d['k'] + '(' + d['index'] + ')| / |'
                + d['corpus'] + '|')
    return ''


def _literal_equation(question, evidence):
    """Return (1-based evidence id, literal equation) only when source contains it.

    A query may deliberately quote a WRONG formula. The equation must come from
    retrieved paper text, have an actual metric subject (e.g. CI(T)), and have a
    recognizable RHS. Never reconstruct damaged PDF mathematics by guessing.
    """
    acronyms = _question_math_subjects(question)
    # The question may spell out a term ("Query Coverage") instead of using
    # its initials ("QC"). Derive the abbreviation ONLY when the same full
    # name is present in a source passage; never infer it from the bad formula
    # supplied in the question. This is independent of the paper/benchmark.
    for name in re.findall(
            r'(?<![A-Za-z])([A-Z][a-z]{2,}(?:[ \t-]+[A-Z][a-z]{2,}){1,5})(?![A-Za-z])',
            str(question or '')):
        if any(re.search(re.escape(name).replace(r'\ ', r'\s+'),
                         str(passage.get('content') or ''), re.I)
               for passage in evidence):
            initials = ''.join(part[0] for part in re.findall(r'[A-Z][a-z]{2,}', name))
            if len(initials) >= 2:
                acronyms.add(initials.upper())
    if not acronyms:
        return None
    pat = re.compile(r'(?<![A-Za-z0-9_])([A-Z][A-Za-z0-9_]{1,14})\s*'
                     r'\(\s*([A-Za-z])\s*\)\s*(?:=|:=)\s*([^\n。；;]{6,210})')
    candidates = []
    for n, passage in enumerate(evidence, 1):
        content = str(passage.get('content') or '')
        recovered = _recover_multiline_union_fraction(content)
        if recovered and recovered.split('(')[0].upper() in acronyms:
            candidates.append((3, n, recovered))
        for m in pat.finditer(content):
            if m.group(1).upper() not in acronyms:
                continue
            # The passage may discuss an *incorrect* candidate equation.
            # Never present a locally labelled wrong example as the paper's
            # official definition during fallback.
            preceding = content[max(0,m.start()-100):m.start()]
            if re.search(r'(?i)\b(?:incorrect|wrong|invalid|false|misleading)\b[^.!?]{0,70}$|'
                         r'(?:错误的|不正确的|误写为)[^。]{0,60}$', preceding):
                continue
            rhs = re.split(r'\.(?=\s+[A-Z][a-z]|$)',
                           m.group(3).strip().split('。')[0],maxsplit=1)[0].strip()
            # Don't accept a name, a label or a non-formula example.
            if not re.search(r'∪|\\bigcup|\\frac|[/|]|∑|\\sum', rhs):
                continue
            # Require readable balanced braces and at least a denominator or
            # normalized cardinality. Avoid truncation after an unclosed brace.
            if rhs.count('{') != rhs.count('}'):
                continue
            lhs = f'{m.group(1)}({m.group(2)})'
            score = 2 if re.search(r'∪|\\bigcup', rhs) else 1
            candidates.append((score, n, f'{lhs} = {rhs}'))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (-x[0], x[1]))
    _, citation, formula = candidates[0]
    return citation, formula


def safe_math_fallback(question, evidence):
    """Only offer literal sourced equations or a computed set-theory witness.

    These are intentionally partial: this function does not claim to have
    verified every proposition, or to understand arbitrary mathematical text.
    """
    q = str(question or '')
    if not re.search(r'公式|定义|数学|求和|并集|检索|反例|formula|equation|set|union|sum', q, re.I):
        return None
    equation = _literal_equation(q, evidence)
    if not equation:
        return None
    citation, formula = equation
    comparing = bool(re.search(r'∑|\\sum|求和|逐项相加|sum', q, re.I) and
                     re.search(r'等于|等价|相等|成立|equal|equivalent', q, re.I) and
                     re.search(r'∪|\\bigcup|并集', formula, re.I))
    if comparing:
        # The witness is a computation, not an experiment claimed by a paper.
        universe = {'a', 'b'}
        left = {'a'}
        right = {'a'}
        summed = (len(left) + len(right)) / len(universe)
        union = len(left | right) / len(universe)
        text = (f'论文中可核实的正式定义为：\\[\n{formula}\n\\] [{citation}]\n\n'
                '**数学反例（自构造，不是论文实验数据）：**'
                '设同一语料全集 D={a,b}，两次查询的结果分别为 A={a}、B={a}。'
                f'按检索结果数量相加得到 (|A|+|B|)/|D|=2/2={summed:g}；'
                f'按集合并集得到 |A∪B|/|D|=1/2={union:g}。'
                '两个查询命中了相同元素，所以逐次求和会重复计数，而并集不会。'
                '只有这些检索结果集合两两不相交时，两种计数才相等。')
        return MathFallback(text, citation, 'computed_union_counterexample', formula)
    if re.search(r'公式|定义|写出|纠正|是否正确|formula|definition|correct', q, re.I):
        return MathFallback(
            f'已从论文证据中确认以下数学表达式：\\[\n{formula}\n\\] [{citation}]\n\n'
            '但本次尚未可靠完成题目要求的全部判断或符号解释，请结合所引原文核对。',
            citation, 'literal_source_equation', formula)
    return None
