"""Normalize and validate skill output *without* pretending to prove facts.

Numeric grounding and real citation mapping remain in app.agent.facts / workflow.
"""
import re
from dataclasses import dataclass

_VERDICTS = {
    'supported': 'supported', 'correct': 'supported', 'true': 'supported', '正确': 'supported',
    'refuted': 'refuted', 'false': 'refuted', 'incorrect': 'refuted', '错误': 'refuted',
    'insufficient_evidence': 'insufficient_evidence', 'insufficient': 'insufficient_evidence',
    'unknown': 'insufficient_evidence', '证据不足': 'insufficient_evidence',
}
_LABELS = {'supported': '正确', 'refuted': '错误',
           'insufficient_evidence': '证据不足'}


@dataclass(frozen=True)
class SkillOutput:
    answer: str
    citations: tuple[int, ...]
    structured: bool = False
    complete: bool = True
    issue: str = ''
    claims_count: int = 0
    variable_symbols: tuple[str, ...] = ()
    formula: str = ''
    formula_well_formed: bool = True


def _citation_ids(values, max_evidence):
    if not isinstance(values, list):
        return ()
    return tuple(dict.fromkeys(v for v in values if type(v) is int and 1 <= v <= max_evidence))


def _formula_issue(formula: str, question: str = '') -> str:
    """Reject obviously broken model-produced math, not try to prove mathematics.

    Qwen can occasionally return {"formula":"T"} with a perfectly plausible
    variables list. The old completeness check treated that as a correct formula.
    The check also catches control characters produced by bad JSON escaping of
    LaTeX (e.g. \f from a single-backslash \\frac).
    """
    formula = formula.strip()
    if len(formula) < 7 or not re.search(r'[A-Za-z\u0370-\u03ff\u4e00-\u9fff]', formula):
        return '公式字段只有单个符号或过短，缺少完整数学表达式'
    if any(c in formula for c in '\x08\x09\x0b\x0c\x0d'):
        return '公式包含异常控制字符，可能是 JSON 中的 LaTeX 反斜杠未正确转义'
    if formula.count('{') != formula.count('}'):
        return '公式 LaTeX 大括号不匹配'
    has_relation = bool(re.search(r'=|≔|≈|≡|\\(?:equiv|approx|coloneqq|propto)\b', formula))
    has_math = bool(re.search(r'\\(?:frac|sum|int|bigcup|cup|prod|sqrt)\b|[∪∑∫÷×*/+−-]', formula))
    if not (has_relation or has_math):
        return '公式缺少等号或数学运算，仅给出了符号名称'
    if re.search(r'数学定义|数学公式|写出.{0,12}公式|mathematical definition', question, re.I) and not has_relation:
        return '题目要求数学定义，但未给出完整等式'
    # Do not require the union operator to occur in the *main* equation: some
    # papers define E as a union separately, then state CC(T)=|E|/|C|.
    return ''


def _render_formula(formula: str, variables, language: str = 'zh') -> tuple[str, tuple[str, ...]]:
    # Local JSON models may use {"x": "meaning"} or {"x": {"meaning": ...}}
    # instead of the documented [{"symbol": "x", "meaning": ...}] array.
    # Never slice a dict: that raises KeyError(slice(None, 16, None)).
    if isinstance(variables, dict):
        variables = [
            {"symbol": symbol,
             "meaning": description.get("meaning") or description.get("description")}
            if isinstance(description, dict) else
            {"symbol": symbol, "meaning": description}
            for symbol, description in list(variables.items())[:16]
        ]
    elif isinstance(variables, list):
        variables = variables[:16]
    else:
        variables = []
    symbols, lines = [], []
    for entry in variables:
        if isinstance(entry, dict) and entry.get('symbol') and entry.get('meaning'):
            symbol = str(entry['symbol']).strip().strip('$')
            symbols.append(symbol)
            lines.append(f'- \\({symbol}\\)：{entry["meaning"]}')
    rendered = f'\\[\n{formula}\n\\]'
    if lines:
        rendered += ('\n\nWhere:\n' if language=='en' else '\n\n其中：\n') + '\n'.join(lines)
    return rendered, tuple(symbols)


def normalize_skill_output(obj, primary: str | None, max_evidence: int, language='zh') -> SkillOutput:
    """Accept model-native skill JSON and previous answer/citations JSON.

    This normalizer never trusts arbitrary model citation IDs: evidence numbers
    must refer to the actual prompt's numbered passages.
    """
    if not isinstance(obj, dict):
        return SkillOutput('', (), issue='无有效 JSON 对象')
    if primary == 'verifyClaims' and isinstance(obj.get('claims'), list) and obj['claims']:
        parts, citation_ids = [], []
        for index, item in enumerate(obj['claims'][:12], 1):
            if not isinstance(item, dict):
                return SkillOutput('', (), issue=f'第 {index} 项不是结构化判断')
            verdict = _VERDICTS.get(str(item.get('verdict', '')).strip().lower())
            assertion = str(item.get('claim') or '').strip()
            explanation = str(item.get('explanation') or '').strip()
            if not (verdict and assertion and explanation):
                return SkillOutput('', (), issue=f'第 {index} 项缺少判断、主张或解释')
            ids = _citation_ids(item.get('evidence_ids', []), max_evidence)
            citation_ids.extend(ids)
            if language == 'zh':
                parts.append(f'{index}. **{_LABELS[verdict]}**：{assertion}。{explanation}' +
                             (' ' + ''.join(f'[{n}]' for n in ids) if ids else ''))
            else:
                parts.append(f'{index}. **{verdict.replace("_", " ")}**: {assertion}. {explanation}' +
                             (' ' + ''.join(f'[{n}]' for n in ids) if ids else ''))
        formula = str(obj.get('formula') or '').strip()
        symbols = ()
        if formula:
            # A composed verifyClaims + explainFormula answer may attach an
            # equation to independently verified claim verdicts.
            math_text, symbols = _render_formula(formula, obj.get('variables') or [], language)
            parts.append(math_text)
            citation_ids.extend(_citation_ids(obj.get('formula_citations', []), max_evidence))
        return SkillOutput('\n\n'.join(parts), tuple(dict.fromkeys(citation_ids)),
                           structured=True, complete=True, claims_count=len(parts)-bool(formula),
                           variable_symbols=symbols, formula=formula,
                           formula_well_formed=not bool(_formula_issue(formula)) if formula else True)
    if primary == 'explainFormula' and isinstance(obj.get('variables'), list) and obj.get('formula'):
        formula = str(obj.get('formula')).strip()
        formula = re.sub(r'^(?:\$\$|\\\[)|(?:\$\$|\\\])$', '', formula).strip()
        text, symbols = _render_formula(formula, obj['variables'], language)
        explanation=str(obj.get('explanation') or '').strip()
        ids=_citation_ids(obj.get('citations'), max_evidence)
        if explanation:
            text+='\n\n'+explanation
        if ids:
            text+=' '+''.join(f'[{n}]' for n in ids)
        return SkillOutput(text, ids, structured=True, complete=bool(symbols and explanation),
                           variable_symbols=symbols, formula=formula,
                           formula_well_formed=not bool(_formula_issue(formula)))
    answer=obj.get('answer')
    ids=_citation_ids(obj.get('citations'),max_evidence)
    return SkillOutput(answer.strip() if isinstance(answer,str) else '', ids)


def validate_skill_output(output: SkillOutput, primary: str | None, question: str) -> str:
    """Cheap structural checks only. Groundedness is checked separately."""
    if output.issue:
        return output.issue
    if not output.answer:
        return '技能输出缺少答案'
    needs_claim_formula = primary == 'verifyClaims' and bool(re.search(
        r'数学公式|数学定义|结合.{0,15}公式|根据.{0,15}公式|'
        r'写出.{0,16}公式|正确公式|公式是否正确|解释.{0,8}符号|'
        r'formula|equation|correct formula', question, re.I))
    if needs_claim_formula and not output.structured:
        # Old answer/citations JSON remains supported, but it must not dodge
        # a requested second task by giving prose-only verdicts.
        if not re.search(r'[A-Za-z][A-Za-z_0-9()]*\s*=\s*[^\s。；]{4,}',
                         output.answer):
            return '逐项判断未提供用户要求的数学等式'
    if primary == 'verifyClaims' and output.structured:
        if not output.claims_count:
            return '未完成逐项判断'
        if re.search(r'(?:三个判断|三项判断|three claims)', question, re.I) and output.claims_count < 3:
            return f'只核验了 {output.claims_count}/3 项判断'
        if re.search(r'(?:两个判断|两项判断|two claims)', question, re.I) and output.claims_count < 2:
            return f'只核验了 {output.claims_count}/2 项判断'
        # Claim verification can also require a mathematical derivation. Do not
        # mark the claims complete while ignoring the explicit formula request.
        if needs_claim_formula:
            if not output.formula:
                return '逐项判断已经生成，但仍缺少用户要求的数学公式'
            problem = _formula_issue(output.formula, question)
            if problem:
                return problem
    if primary == 'explainFormula' and output.structured:
        if not output.complete:
            return '公式说明缺少变量定义或解释'
        # Explicitly named variables must have their own explanation, not merely
        # appear inside the equation. Do not infer undeclared variable names.
        if re.search(r'变量|符号|each variable|each symbol', question, re.I):
            named=re.findall(
                r'(?:^|[、，,\s])([A-Za-z](?:_[A-Za-z0-9]+)?(?:\([a-z]\))?|τ)'
                r'(?=[、，,\s]|和|及|$)',question)
            missing=[symbol for symbol in dict.fromkeys(named)
                     if symbol not in output.variable_symbols]
            if missing:
                return '未分别解释指定变量：'+', '.join(missing)
        problem = _formula_issue(output.formula, question)
        if problem:
            return problem
    return ''
