"""Lightweight, source-aware answer quality checks for the demo Agent.

These checks catch a few high-impact *contradictions*. They are not a theorem
prover, a full NLI model or a substitute for reviewing the cited paper.
No paper-specific terms, numbers, or reference answers live in this module.
"""
import re
import unicodedata
from decimal import Decimal, InvalidOperation


def _plain(text):
    return unicodedata.normalize('NFKC', str(text or '')).replace('\\left', '').replace('\\right', '')


def _equation_lhs(text):
    """Read a short LHS like coverage(T) or CC(T), not an entire equation."""
    m = re.search(r'(?<![\w\\])([A-Za-z][A-Za-z0-9_]{0,20}(?:\s*\([A-Za-z][A-Za-z0-9_]*\))?)\s*(?:=|:=|≡)',
                  _plain(text))
    return re.sub(r'\s+', '', m.group(1)) if m else ''


def _named_subjects(question):
    q = _plain(question)
    acronyms = set(re.findall(r'(?<![A-Za-z])[A-Z]{2,10}(?![A-Za-z])', q))
    for phrase in re.findall(r'\b[A-Z][a-z]+(?:[ -][A-Z][a-z]+){1,4}\b', q):
        acronyms.add(''.join(word[0] for word in phrase.replace('-', ' ').split()))
    return acronyms


def _source_math_lhs(question, evidence):
    """Use an equation literally present in cited evidence as a subject anchor.

    PDF extraction can lose symbols on the RHS. Matching only the LHS avoids
    pretending we parsed the full mathematics from broken PDF text.
    """
    candidates = _named_subjects(question)
    found = set()
    for item in evidence:
        content = _plain(item.get('content', ''))
        for m in re.finditer(r'(?m)(?<!\w)([A-Za-z][A-Za-z0-9_]{1,18}(?:\s*\([A-Za-z]\))?)\s*(?:=|:=)', content):
            lhs = re.sub(r'\s+', '', m.group(1))
            if lhs.split('(')[0].upper() in candidates:
                found.add(lhs)
    return next(iter(sorted(found, key=len, reverse=True)), '')


def formula_grounding_issue(question, formula, evidence, answer=''):
    """Check source-defined formula *subject* and explicit math obligations.

    This deliberately does not claim that syntactically plausible algebra is
    mathematically equivalent to the paper's definition.
    """
    q = _plain(question)
    expr = _plain(formula).strip()
    if not expr:
        # A legacy prose answer is subject to older completeness checks; do
        # not try to parse LaTeX embedded inside arbitrary Chinese text.
        return ''
    expected = _source_math_lhs(q, evidence)
    got = _equation_lhs(expr)
    if expected and got and expected.split('(')[0].upper() != got.split('(')[0].upper():
        return f'公式左侧 {got} 与原文定义的 {expected} 不一致，可能混淆了测试集合和指标'
    needs_set_definition = bool(re.search(r'公式|定义|equation|formula', q, re.I)) and (
        ('∪' in q or r'\bigcup' in q or '并集' in q) and
        any(re.search(r'∪|\\bigcup|\bunion\b', _plain(ev.get('content', '')), re.I)
            for ev in evidence)
    )
    if needs_set_definition and not re.search(r'\\bigcup|∪', expr):
        return '原文使用集合并集，但给出的形式化定义没有包含并集运算'
    if needs_set_definition and re.search(r'\\bigcup|∪', expr) and not re.search(
            r'(?:q\s*\\in\s*\w+|q\s*∈\s*\w+|\\bigcup_\s*\{[^}]+\}|∪_\s*\{[^}]+\})', expr):
        return '并集公式缺少明确的索引范围，无法判断是跨哪些查询或元素取并集'
    return ''


def requests_counterexample(question):
    """Recognize constructive refutations even when a user says 'example', not 'counterexample'.

    An ordinary request for an illustrative example is NOT a proof obligation.
    Require an explicit mathematical equivalence/universality question as well
    as a request to construct a concrete example. This is domain-independent.
    """
    q = _plain(question)
    if re.search(r'反例|counterexample', q, re.I):
        return True
    construction = re.search(
        r'(?:构造|举出|举个?|给出|提供|设计|用).{0,55}(?:例子|示例|实例|例|example)|'
        r'(?:construct|give|provide|show).{0,55}(?:example|case)', q, re.I)
    challenge = re.search(
        r'是否.{0,18}(?:总|始终|恒|相等|等于|等价|成立)|'
        r'(?:总是|始终|恒等|一定).{0,20}(?:等于|相等|成立)|'
        r'(?:always|necessarily).{0,30}(?:equal|equivalent|hold)', q, re.I)
    mathematical = re.search(r'∑|∪|\\sum|\\bigcup|求和|并集|集合|公式|等价|equation|set|union|sum', q, re.I)
    return bool(construction and challenge and mathematical)


def counterexample_issue(question, answer):
    """Check basic constructive-proof obligations without hardcoding a paper.

    Do not attempt to prove arbitrary mathematics; check that an actual pair
    of mathematical objects, the distinguishing overlap, and numerical
    comparisons were supplied. Source definitions must still be verified by
    the regular groundedness checks.
    """
    if not requests_counterexample(question):
        return ''
    q = _plain(question)
    if not re.search(r'并集|求和|重复|集合|∪|∑|\\bigcup|\\sum|sum|union|set|等于', q, re.I):
        return ''
    a = _plain(answer)
    # Accept both {x} and LaTeX \{x\}; capture explicitly assigned result
    # sets when present, rather than comparing the corpus C with a result set.
    assigned = []
    pattern = (r'(?P<name>[A-Za-z][A-Za-z_0-9]*(?:\([^)]*\))?)\s*=\s*'
               r'\\?\{(?P<items>[^{}]{1,100})\\?\}')
    for match in re.finditer(pattern, a):
        name = match.group('name').strip()
        # Corpus/universe notation is not one of the retrieved result sets.
        if name.casefold() in ('c', 'corpus', 'universe', 'omega'):
            continue
        assigned.append({x.strip().replace('\\', '') for x in
                         re.split(r'[,，]', match.group('items')) if x.strip()})
    all_sets = re.findall(r'\\?\{([^{}]{1,100})\\?\}', a)
    parsed = (assigned if len(assigned) >= 2 else [
        {x.strip().replace('\\', '') for x in re.split(r'[,，]', v) if x.strip()}
        for v in all_sets])
    if len(parsed) < 2:
        return '数学反例必须明确列出至少两个实际集合，不能只描述两个自然语言问题'
    if re.search(r'相同|重复|重叠|并集|求和|union|overlap|sum', q, re.I):
        if not any(parsed[i] & parsed[j] for i in range(len(parsed)) for j in range(i+1,len(parsed))):
            return '构造的两个检索结果集合没有共同元素，无法证明重复计数'
    # Both the incorrect and the correct calculation must be visible.
    # Support normal fractions and LaTeX \frac{n}{d}; an example can use
    # symbolic cardinalities instead of a denominator when the user allows it.
    numeric_fractions = re.findall(r'\d+\s*[/÷]\s*\d+', a)
    latex_fractions = re.findall(r'\\frac\s*\{\s*\d+\s*\}\s*\{\s*\d+\s*\}', a)
    count_results = re.findall(r'(?:求和|并集|\\sum|\\bigcup|∑|∪|sum|union|\|[^|\n]{1,60}\|)'
                               r'[^=\n]{0,55}=\s*\d+(?:\.\d+)?', a, re.I)
    if len(numeric_fractions) + len(latex_fractions) < 2 and len(count_results) < 2:
        return '数学反例必须计算两种定义的数值（求和与并集），不能只给出真假结论'
    for found in re.finditer(r'(\d+)\s*[/÷]\s*(\d+)\s*(?:=|≈)\s*(\d+(?:\.\d+)?)', a):
        numerator, denominator, claimed = map(Decimal, found.groups())
        if denominator == 0 or abs(numerator / denominator - claimed) > Decimal('0.025'):
            return '反例中的分数计算不成立，必须重新核对具体数字'
    return ''


def _number(value):
    try:
        return Decimal(str(value).strip().replace(',', '').rstrip('×x%'))
    except (InvalidOperation, ValueError):
        return None


def universal_table_issue(question, structured_answer, plain_answer, tables):
    """Refute a universal higher-is-better claim when a sourced counterexample exists.

    A single counterexample is sufficient to refute "ALL rows", but the absence
    of an observed counterexample is not proof of a universal claim.
    """
    q = _plain(question)
    if not (re.search(r'所有|全部|每个|均|all|every|always', q, re.I)
            and re.search(r'APFD', q, re.I)):
        return ''
    counterexamples = []
    for table in tables:
        if not {'CC', 'R'}.issubset(set(table.get('headers', []))):
            continue
        for row in table.get('rows', []):
            cc, rand = _number(row.get('CC')), _number(row.get('R'))
            if cc is not None and rand is not None and cc <= rand:
                metric = str(row.get('RAG Eval. Metric') or row.get(table['headers'][1], '')).strip()
                counterexamples.append((metric, cc, rand))
    if not counterexamples:
        return ''
    # Do not require every counterexample to be listed: one suffices.
    a = _plain(plain_answer)
    has_counterexample = any(
        (str(cc) in a and str(rand) in a and (not metric or metric.casefold() in a.casefold()))
        for metric, cc, rand in counterexamples)
    if not has_counterexample:
        return '论文表格存在全称判断的数值反例，但回答没有给出对应指标和两侧数值'
    if isinstance(structured_answer, dict) and isinstance(structured_answer.get('claims'), list):
        for item in structured_answer['claims']:
            if not isinstance(item, dict):
                continue
            assertion = str(item.get('claim') or '')
            if (re.search(r'所有|全部|每个|均|all|every|always', assertion, re.I)
                    and re.search(r'APFD|指标|metrics?', assertion, re.I)
                    and str(item.get('verdict', '')).casefold() in ('supported', 'correct', 'true', '正确')):
                return '全称 APFD 主张存在反例，不能将该项判断为 supported'
    # For old free-text JSON, check for an explicit negation before accepting.
    if not isinstance(structured_answer, dict) or not isinstance(structured_answer.get('claims'), list):
        if not re.search(r'错误|不成立|并非所有|不是所有|不一定|不全|反例|refuted|false|incorrect|not all', a, re.I):
            return '检测到 APFD 全称判断的反例，但没有明确否定该主张'
    return ''


def experiment_followup_issue(question, answer, evidence):
    """Verify a requested follow-up experiment is actually discussed, not substituted
    with a restatement of the original result table.
    """
    q = _plain(question)
    if not re.search(r'用什么实验|什么实验解释|如何实验|补充实验|额外实验|验证这一例外|'
                     r'what experiment|follow.up experiment|supplementary experiment',q,re.I):
        return ''
    sources = '\n'.join(_plain(x.get('content','')) for x in evidence)
    if not re.search(r'\brepeat(?:ed)?\b|\bsupplementary\b|\badditional experiment\b|'
                     r'重新实验|补充实验|复现实验', sources, re.I):
        return ''  # Without the relevant original paragraph, avoid invented demands.
    a = _plain(answer)
    if not re.search(r'重新|重复|复做|补充实验|再次|更强|repeated|supplementary|reran|stronger',a,re.I):
        return '相关原文描述了补充/重复实验，但答案只复述了原始实验结果'
    # If source explicitly identifies a model used in that follow-up, including
    # its name distinguishes the actual experimental change from a vague guess.
    # Prefer the model explicitly named as the *new experimental setup*.
    # Merely repeating the dataset identifier is not a valid explanation.
    named_model=re.search(
        r'(?i)(?:stronger|different|new|alternative)\s+(?:query\s+)?'
        r'(?:generator|model)[^\n]{0,90}\(([^)]+)\)',sources)
    if named_model:
        model_name=named_model.group(1).strip()
        if model_name and model_name.casefold() not in a.casefold():
            return '补充实验明确指定了模型/配置，但答案没有交代该实验条件'
    return ''


def formal_equation_issue(question, answer, evidence, formula=''):
    """Extra gate for requests to *write/correct the formal definition*.

    Generic 'CC is exercised chunks / total' prose may communicate intuition,
    but it does not satisfy a request for the indexed definition in the paper.
    """
    q = _plain(question)
    if not re.search(r'写出.{0,25}(?:正确)?公式|正确公式|正式.{0,25}公式|形式化定义|'
                     r'correct formula|formal definition',q,re.I):
        return ''
    expression = formula
    if not expression:
        candidates = re.findall(r'\\\[\s*(.*?)\s*\\\]|\$\$\s*(.*?)\s*\$\$', answer,re.S)
        expression = (next((a or b for a,b in reversed(candidates)), '')
                      if candidates else answer)
    # A prose ratio can be a correct intuition, but a request to *write the
    # formal equation* explicitly also requires a symbolic relationship.
    if not _equation_lhs(expression):
        return '题目要求正式数学等式，回答只给出文字比例或单个符号'
    expected = _source_math_lhs(q,evidence)
    if expected:
        lhs = _equation_lhs(expression)
        if not lhs or lhs.split('(')[0].upper()!=expected.split('(')[0].upper():
            return f'没有写出原文中以 {expected} 为左侧的正式数学定义'
        if '(' in expected and '(' not in lhs:
            return f'正式定义遗漏原文指标的参数：{expected}'
    source_math='\n'.join(_plain(x.get('content','')) for x in evidence)
    if (re.search(r'q\s*∈\s*[A-Z]|q\s*\\in\s*[A-Z]',source_math,re.I)
            and ('∪' in source_math or '\\bigcup' in source_math or '∪' in q)):
        if not re.search(r'q\s*(?:\\in|∈)\s*[A-Za-z]',expression):
            return '原文数学定义通过查询集合索引取并集，回答遗漏了该索引关系'
    return ''


def recover_embedded_formula(question, structured, evidence):
    """Recover a formal equation written in a claim explanation instead of `formula`.

    Local 4B models sometimes put the full correct equation into a claim's
    explanation while omitting the sibling formula field. Require the source's
    literal LHS, so the incorrect equation quoted *by the question* cannot be
    mistaken for the requested definition. Other semantic checks still apply.
    """
    if not isinstance(structured, dict) or structured.get('formula'):
        return ''
    expected = _source_math_lhs(question, evidence)
    if not expected or '(' not in expected:
        return ''
    for claim in structured.get('claims') or []:
        if not isinstance(claim, dict):
            continue
        explanation = _plain(claim.get('explanation', ''))
        match = re.search(r'(?<![\w])' + re.escape(expected) + r'\s*=\s*([^\n。；;]+)', explanation, re.I)
        if not match:
            continue
        rhs = re.split(r'[,，]\s*(?:where|其中|with\s+|这里)', match.group(1), maxsplit=1, flags=re.I)[0]
        equation = expected + ' = ' + rhs.strip().rstrip('.,， ')
        if len(equation) >= 12 and ('|' in equation or r'\frac' in equation):
            return equation
    return ''


def aggregate_table_verdict_issue(question, structured, sources):
    """Check simple numeric aggregate verdicts against a labelled source row.

    Only covers *atomic* claims explicitly saying an average speedup equals a
    number. A table's Average row cannot prove arbitrary causal or compound
    claims, so ambiguous or multi-clause assertions are not classified.
    """
    if not isinstance(structured, dict) or not isinstance(structured.get('claims'), list):
        return ''
    for claim in structured['claims']:
        if not isinstance(claim, dict):
            continue
        assertion = _plain(claim.get('claim', ''))
        if not (re.search(r'平均|average|mean', assertion, re.I)
                and re.search(r'加速|速度提升|speedup|faster', assertion, re.I)):
            continue
        if re.search(r'因此|所以|意味着|导致|从而|implies|therefore|hence|always|所有|全部', assertion, re.I):
            continue
        value_match = re.search(r'(?<![\d.])(\d+(?:\.\d+)?)\s*(?:×|倍|x\b)', assertion, re.I)
        if not value_match:
            continue
        claimed = _number(value_match.group(1))
        if claimed is None:
            continue
        verified_values = []
        for table in sources:
            headers = table.get('headers') or []
            speedup = [h for h in headers if 'speedup' in str(h).lower() or '加速比' in str(h)]
            if ('random' in assertion.lower() or '随机' in assertion) and any('(R)' in h for h in speedup):
                speedup = [h for h in speedup if '(R)' in h]
            if len(speedup) != 1:
                continue  # unclear which aggregate measure the claim denotes
            header = speedup[0]
            for row in table.get('rows') or []:
                if any(str(row.get(h, '')).strip().casefold() in ('average', 'avg', '平均')
                       for h in headers[:2]):
                    actual = _number(row.get(header))
                    if actual is not None:
                        verified_values.append(actual)
        if len(set(verified_values)) != 1:
            continue
        actual = verified_values[0]
        verdict = str(claim.get('verdict', '')).casefold()
        if claimed == actual and verdict in ('refuted', 'incorrect', 'false', '错误'):
            return '原始表格 Average 行确认平均数值与主张一致，但回答将该原子主张误判为错误'
        if claimed != actual and verdict in ('supported', 'correct', 'true', '正确'):
            return '主张给出的平均加速比与表格 Average 行数值不同，不能标记为正确'
    return ''


def matching_experiment_citation(answer, evidence):
    """Find a citable source for a described follow-up *experimental change*.

    This is not a general entailment checker. To automatically attach a missing
    citation, both the answer and a follow-up experiment source must share an
    identifiable named experimental configuration/model, not merely keywords
    like 'experiment' or 'result'.
    """
    text = _plain(answer)
    if not re.search(r'重新|重复|补充|更强|再次|repeat|stronger|reran|supplementary', text, re.I):
        return None
    # Prefer the named model/configuration in the FOLLOW-UP setup, not a
    # dataset ID or an earlier baseline model mentioned elsewhere in the chunk.
    # A strict match reduces false automatic citation repairs.
    changed_model = re.compile(
        r'(?is)\b(?:stronger|different|alternative|new|improved)\s+'
        r'(?:(?:query|test|generation)\s+)?(?:generator|model)\s*'
        r'\(\s*([A-Za-z][A-Za-z0-9]*(?:[-_.][A-Za-z0-9]+)+)\s*\)')
    for index, item in enumerate(evidence, 1):
        content = _plain(item.get('content', ''))
        if not re.search(r'(?i)repeat(?:ed)?|supplementary experiment|additional experiment|'
                         r'更强.{0,20}实验|补充实验', content):
            continue
        model_names={m.group(1).casefold() for m in changed_model.finditer(content)}
        # Unicode \w includes CJK characters. E.g. 'Model-5生成' must still
        # match the ASCII model identifier 'Model-5'.
        if any(re.search(r'(?i)(?<![A-Za-z0-9_-])' + re.escape(name)
                         + r'(?![A-Za-z0-9_-])', text) for name in model_names):
            return index
    return None


def reconcile_source_aggregate_claims(structured, sources):
    """Correct an unambiguous *atomic numeric equality* using a table Average row.

    Only rewrites an aggregate verdict when both the aggregate's meaning and
    numeric value are explicit in the actual source. It never edits broader
    logical/causal assertions or infers an average from individual rounded rows.
    Returns (possibly corrected JSON object, count of corrected claim verdicts).
    """
    if not isinstance(structured, dict) or not isinstance(structured.get('claims'), list):
        return structured, 0
    updated = dict(structured)
    updated_claims = []
    count = 0
    for claim in structured['claims']:
        if not isinstance(claim, dict):
            updated_claims.append(claim)
            continue
        assertion = _plain(claim.get('claim', ''))
        if not (re.search(r'平均|average|mean', assertion, re.I)
                and re.search(r'加速|速度提升|speedup|faster', assertion, re.I)):
            updated_claims.append(claim)
            continue
        if re.search(r'因此|所以|意味着|导致|从而|implies|therefore|hence|always|所有|全部', assertion, re.I):
            updated_claims.append(claim)
            continue
        claimed_match = re.search(r'(?<![\d.])(\d+(?:\.\d+)?)\s*(?:×|倍|x\b)', assertion, re.I)
        if not claimed_match:
            updated_claims.append(claim)
            continue
        claimed = _number(claimed_match.group(1))
        candidates = []
        for table in sources:
            headers = table.get('headers') or []
            speedup = [h for h in headers if 'speedup' in str(h).lower() or '加速比' in str(h)]
            if ('random' in assertion.lower() or '随机' in assertion) and any('(R)' in h for h in speedup):
                speedup = [h for h in speedup if '(R)' in h]
            if len(speedup) != 1:
                continue
            header = speedup[0]
            for row in table.get('rows') or []:
                if any(str(row.get(h, '')).strip().casefold() in ('average', 'avg', '平均')
                       for h in headers[:2]):
                    actual = _number(row.get(header))
                    if actual is not None:
                        candidates.append((actual, table, header, row.get(header)))
        # Multiple source averages could refer to different setups. Abstain.
        if len(candidates) != 1 or claimed is None:
            updated_claims.append(claim)
            continue
        actual, table, header, raw = candidates[0]
        expected = 'supported' if claimed == actual else 'refuted'
        verdict = str(claim.get('verdict', '')).casefold()
        normalized = ('supported' if verdict in ('supported', 'correct', 'true', '正确') else
                      'refuted' if verdict in ('refuted', 'incorrect', 'false', '错误') else None)
        if normalized is not None and normalized != expected:
            replacement = dict(claim, verdict=expected)
            replacement['explanation'] = (
                f'Table {table["table_number"]} 的 Average 行记录 {header}={raw}，'
                + ('与这一平均加速比主张一致。' if expected == 'supported'
                   else f'与主张中的 {claimed_match.group(0)} 不一致。'))
            updated_claims.append(replacement)
            count += 1
        else:
            updated_claims.append(claim)
    updated['claims'] = updated_claims
    return updated, count
