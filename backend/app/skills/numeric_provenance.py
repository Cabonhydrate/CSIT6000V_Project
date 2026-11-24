"""Conservative numeric provenance checks for paper-grounded answers.

This module distinguishes three *uses* of decimal/percentage numbers:
reported evidence, simple derivations, and explicitly illustrative examples.
It is not a semantic fact-checker or a general symbolic algebra system.
"""
from __future__ import annotations

import ast
from decimal import Decimal, InvalidOperation
import re
import unicodedata

_PRECISE = re.compile(
    r'(?<![A-Za-z0-9])(?:\d+(?:\.\d+)?\s*%|\d+\.\d+)(?![A-Za-z0-9])'
)
_CITATION = re.compile(r'\[\d+\]')
_HYPOTHETICAL = re.compile(
    r'假设|假定|假如|如果|(?:^|[，,:：\s])若|设想|'
    r'(?i:hypothetical|suppos(?:e|ing)|assuming|let\s+us\s+assume|for\s+example\s+if)'
)
# Require both an example frame and a modal, not merely "may" in an
# empirical claim: "For example, a test suite might cover 90%" is a
# scenario; "the experiment may have reached 90%" is not evidence.
_MODAL_EXAMPLE = re.compile(
    r'(?i:(?:for\s+example|e\.g\.|imagine|consider|as\s+an\s+illustration)'
    r'[^.!?\n]{0,130}\b(?:might|could|would)\b)'
)
_ENGLISH_SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
_EXAMPLE = re.compile(r'例如|举例|举个例子|比如|示例|(?i:for\s+example)')
_FACT_ASSERTION = re.compile(
    r'(?:论文|作者|研究|实验|(?:Table|表格|表)\s*\d+|数据集|'
    r'(?i:paper|study|experiment|results|table))'
    r'.{0,55}?(?:报告|表明|显示|指出|达到|获得|得到|'
    r'(?i:reports?|found|shows?|achieved|yields?))'
)
# A full computation before a result, e.g. (0.839-0.669)/0.669 ≈ 0.2541
# or (0.839-0.669)/0.669 × 100% ≈ 25.41%.
_ARITHMETIC = re.compile(
    r'(?P<expr>\(?\d[\d\s.%()+\-*/×÷−－]*[\d%)）])\s*'
    r'(?:≈|≃|~=|=|等于|约为|约等于)\s*'
    r'(?P<result>\d+(?:\.\d+)?\s*%?)'
)
_CC_PERCENT = re.compile(r'(?i:CC|覆盖率)\s*(?:=|为|是|达到|约为)\s*(\d+(?:\.\d+)?)\s*%')
_TOTAL = re.compile(r'(?:共有|总共有|总计|总共|语料库(?:中)?有|文档库(?:中)?有|有)\s*(\d+)\s*(?:个|块|条|chunks?)')
_HITS = re.compile(r'(?:检索到|覆盖了?|命中了?|检索了?|取到)\s*(\d+)\s*(?:个|块|条|chunks?)')


def _norm(text: str) -> str:
    return unicodedata.normalize('NFKC', str(text or '')).replace('−', '-').replace('－', '-')


def _has_literal(text: str, token: str) -> bool:
    token = re.sub(r'\s+', '', token)
    # A number is not sourced merely because its digits occur inside another.
    normalized = re.sub(r'(?<=\d)\s+%', '%', text)
    return bool(re.search(r'(?<![A-Za-z0-9_.])'+re.escape(token)+
                          r'(?![A-Za-z0-9_%]|\.(?=\d))', normalized))


def _sentence(text: str, position: int) -> tuple[str, int]:
    """Return a local claim/sentence. A hypothetical marker cannot waive later claims."""
    left = max((text.rfind(mark, 0, position) for mark in ('。', '！', '？', '\n')), default=-1)
    rights = [text.find(mark, position) for mark in ('。', '！', '？', '\n')]
    right = min((r for r in rights if r >= 0), default=len(text))
    # English claims need separate sentences, or an illustrative number in
    # sentence A might accidentally license a reported score in sentence B.
    for match in _ENGLISH_SENTENCE_BOUNDARY.finditer(text):
        if match.end() <= position and match.end()-1 > left:
            left = match.end()-1
        if match.start() >= position and match.start() < right:
            right = match.start()
            break
    return text[left+1:right], left+1


def _illustrative(sentence: str) -> bool:
    # "例如，实验中..." alone is NOT a hypothetical result. '例如，若...' is.
    return bool(_HYPOTHETICAL.search(sentence) or _MODAL_EXAMPLE.search(sentence))


def _reported(sentence: str) -> bool:
    return bool(_FACT_ASSERTION.search(sentence) or re.search(
        r'(?:论文|实验|研究|作者|(?i:paper|experiment|study))'
        r'.{0,35}(?:结果|数据|指标|分数|得分|准确率|(?i:results?|scores?|accuracy))',
        sentence))


def _safe_eval(expression: str) -> Decimal | None:
    """Arithmetic-only AST evaluator; no names, calls, attributes or Python eval."""
    expr = _norm(expression).replace('×', '*').replace('÷', '/')
    expr = re.sub(r'(\d+(?:\.\d+)?)\s*%', r'(\1/100)', expr)
    if len(expr) > 140:
        return None
    try:
        node = ast.parse(expr, mode='eval')
        def calculate(n):
            if isinstance(n, ast.Expression):
                return calculate(n.body)
            if isinstance(n, ast.Constant) and type(n.value) in (int, float):
                return Decimal(str(n.value))
            if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd,ast.USub)):
                value=calculate(n.operand)
                return value if isinstance(n.op,ast.UAdd) else -value
            if isinstance(n, ast.BinOp) and isinstance(n.op,(ast.Add,ast.Sub,ast.Mult,ast.Div)):
                lhs,rhs=calculate(n.left),calculate(n.right)
                if isinstance(n.op,ast.Add): return lhs+rhs
                if isinstance(n.op,ast.Sub): return lhs-rhs
                if isinstance(n.op,ast.Mult): return lhs*rhs
                if rhs==0: raise ZeroDivisionError
                return lhs/rhs
            raise ValueError('unsupported operation')
        return calculate(node)
    except (SyntaxError,ValueError,ArithmeticError,InvalidOperation,TypeError):
        return None


def _calculation_for_token(sentence: str, token: str, position: int):
    """Match an explicit nearby expression with this token as the result."""
    for match in _ARITHMETIC.finditer(sentence):
        result=match.group('result').replace(' ','')
        if result != token.replace(' ',''):
            continue
        if not (match.start('result') <= position <= match.end('result')):
            continue
        expr=match.group('expr')
        # A plain literal such as CC=60% is not a derivation.
        if not re.search(r'[/÷+*×−-]',expr):
            continue
        actual=_safe_eval(expr)
        if actual is not None:
            expected=Decimal(result.rstrip('%').strip())
            # Percent can be written in either consistent convention:
            #   fraction ≈ 0.2541  / fraction × 100% ≈ 25.41%
            #   fraction × 100 ≈ 25.41%
            if result.endswith('%') and not re.search(r'[×*]\s*100\s*$',expr):
                expected/=100
            # Reporting rounded percentages or decimal factors is allowed.
            tolerance=(Decimal('0.5')*(Decimal(10)**(-len(result.rstrip('%').split('.')[-1]))))
            if '.' not in result:
                tolerance=Decimal('0.5')
            if result.endswith('%') and not re.search(r'[×*]\s*100\s*$',expr):
                tolerance/=100
            return expr, actual, expected, tolerance
    return None


def _simple_hypothetical_percentage(sentence: str, token: str) -> bool | None:
    """Check common '10 chunks in corpus, 6 hit, CC=60%' worked examples.

    None means no unambiguous calculation was present; this is not a reason
    to claim that an arbitrary answer has been arithmetically verified.
    """
    match = _CC_PERCENT.search(sentence)
    if not match or match.group(1)+'%' != token:
        return None
    totals = _TOTAL.findall(sentence)
    hits = _HITS.findall(sentence)
    if len(totals)!=1 or len(hits)!=1:
        return None
    total,hit=int(totals[0]),int(hits[0])
    if total<=0 or hit>total:
        return False
    return abs(Decimal(hit)/Decimal(total)*100-Decimal(match.group(1))) <= Decimal('0.05')


def audit_numeric_provenance(question: str, answer: str, evidence: list[dict]):
    """Return (audited tokens, first failure message).

    `kind` is one of paper_fact, question_input, calculated, hypothetical.
    An illustrative fact still needs valid arithmetic when the text supplies
    enough operands to verify it. Unknown derivations are not silently accepted.
    """
    if not isinstance(answer,str) or not evidence:
        return [],''
    question=_norm(question)
    source=_norm('\n'.join(str(e.get('content') or '') for e in evidence))
    clean=_norm(_CITATION.sub('',answer))
    records=[]
    for match in _PRECISE.finditer(clean):
        value=match.group().replace(' ','')
        sentence, sentence_start=_sentence(clean,match.start())
        local_pos=match.start()-sentence_start
        previous = ''
        if sentence_start>0 and clean[sentence_start-1]=='。':
            previous,_=_sentence(clean,sentence_start-2)
            previous=previous[-180:]
        entry={'value':value,'kind':'paper_fact','verified':False}
        if _has_literal(source,value):
            entry.update(kind='paper_fact',verified=True)
        elif _has_literal(question,value) and not _reported(sentence[:local_pos+len(value)]):
            # A number repeated from the question is a user input, not proof
            # that the paper actually measured/reported that number.
            entry.update(kind='question_input',verified=True)
        else:
            # An example can span two adjacent sentences: first declare the
            # synthetic data, then compute the result. Never propagate across
            # a paragraph break or into a statement reporting paper results.
            hypothetical=(_illustrative(sentence) or
                          (_illustrative(previous) and not _reported(sentence)))
            # The 100% in 'x * 100% = y%' is a unit conversion, not an
            # empirical 100%-coverage claim. It never bypasses a 100% claim
            # outside of an explicit arithmetic expression.
            conversion_factor = any(
                m.start('expr') <= local_pos < m.end('expr') and value=='100%'
                for m in _ARITHMETIC.finditer(sentence))
            if conversion_factor:
                entry.update(kind='calculated',verified=True)
                records.append(entry)
                continue
            calc=_calculation_for_token(sentence,value,local_pos)
            if calc is not None:
                expr,actual,expected,tolerance=calc
                if abs(actual-expected)>tolerance:
                    entry.update(kind='calculated',verified=False)
                    records.append(entry)
                    return records,f'计算示例 {value} 与算式 {expr.strip()} 的结果不一致'
                operands=re.findall(r'\d+(?:\.\d+)?\s*%?',expr)
                grounded=all(_has_literal(question,n.strip()) or _has_literal(source,n.strip())
                             or n.replace(' ','')=='100%'
                             or (n.strip()=='100' and expr.rstrip().endswith(('* 100','*100','× 100','×100'))
                                 and value.endswith('%')) for n in operands)
                if hypothetical or grounded:
                    entry.update(kind='calculated' if not hypothetical else 'hypothetical',verified=True)
                else:
                    records.append(entry)
                    return records,f'计算结果 {value} 的输入数字未见于问题或所选原文证据'
            elif hypothetical and not _reported(sentence):
                check=_simple_hypothetical_percentage(
                    previous+'。'+sentence if previous and not _illustrative(sentence)
                    else sentence,value)
                if check is False:
                    records.append(entry)
                    return records,f'假设例子中的 {value} 与给定数量不一致'
                entry.update(kind='hypothetical',verified=True)
            else:
                records.append(entry)
                return records,f'精确数值 {value} 未见于问题或所选原文证据，请删除或提供可验证来源'
        records.append(entry)
    return records,''
