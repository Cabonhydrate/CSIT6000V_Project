"""Small file-backed skill registry: no plugin service or additional LLM call.

Business logic lives in the Agent; files here specify answer contracts, not facts.
"""
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .quality import requests_counterexample

_ROOT = Path(__file__).parent
_SKILL_PATHS = {
    'verifyClaims': 'verify_claims/SKILL.md',
    'explainFormula': 'explain_formula/SKILL.md',
    'explainForBeginner': 'explain_for_beginner/SKILL.md',
    'explainBackgroundConcept': 'explain_background_concept/SKILL.md',
    'constructCounterexample': 'construct_counterexample/SKILL.md',
    'guidePaperReading': 'guide_paper_reading/SKILL.md',
    'comparePapers': 'compare_papers/SKILL.md',
    'suggestLearningPath': 'suggest_learning_path/SKILL.md',
}


@dataclass(frozen=True)
class SkillPlan:
    primary: str | None
    styles: tuple[str, ...] = ()
    # A question may require a second *task*, not just a presentation style.
    # Keep styles as the second field so existing SkillPlan(None, styles) works.
    tasks: tuple[str, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        return ((self.primary,) if self.primary else ()) + self.tasks + self.styles


def select_answer_skills(intent: str, *, teaching: bool, level: str = 'beginner',
                         question: str = '') -> SkillPlan:
    """A primary task skill plus an optional, orthogonal teaching-style skill."""
    primary = {'claim_check': 'verifyClaims', 'formula': 'explainFormula'}.get(intent)
    styles = ('explainForBeginner',) if teaching and level == 'beginner' else ()
    # An assertion about a mathematical definition needs *both* verdict and
    # formula. A single intent label must not silently drop half the request.
    # This is task composition, not a hardcoded paper/acronym rule.
    tasks = ('explainFormula',) if (
        primary == 'verifyClaims' and
        re.search(r'数学公式|数学定义|结合.{0,15}公式|根据.{0,15}公式|'
                  r'写出.{0,16}公式|正确公式|公式是否正确|解释.{0,8}符号|'
                  r'formula|equation|mathematical definition|correct formula', question, re.I)
    ) else ()
    # Counterexample construction is a distinct output obligation: a bare
    # true/false verdict or an analogy is not a mathematical counterexample.
    if (primary in ('explainFormula', 'verifyClaims') and
            requests_counterexample(question)):
        tasks += ('constructCounterexample',)
    return SkillPlan(primary, styles, tasks)


@dataclass(frozen=True)
class SkillDefinition:
    name: str
    type: str
    description: str
    output_contract: str
    version: int
    instructions: str


@lru_cache(maxsize=16)
def get_skill_definition(name: str) -> SkillDefinition:
    """Load and validate one SKILL.md manifest; reject malformed Skill files.

    Simple flat YAML-style frontmatter avoids an unnecessary YAML dependency.
    The model only sees the instructions, not the administrative metadata.
    """
    if name not in _SKILL_PATHS:
        raise ValueError(f'Unknown skill: {name}')
    content = (_ROOT / _SKILL_PATHS[name]).read_text(encoding='utf-8').strip()
    parts = content.split('---', 2)
    if len(parts) != 3 or parts[0].strip():
        raise ValueError(f'Skill {name} must have YAML frontmatter')
    fields = {}
    for line in parts[1].splitlines():
        if not line.strip():
            continue
        if ':' not in line:
            raise ValueError(f'Skill {name} has invalid manifest field')
        key, value = line.split(':', 1)
        fields[key.strip()] = value.strip()
    expected = ('name', 'type', 'description', 'output_contract', 'version')
    if any(not fields.get(field) for field in expected) or fields['name'] != name:
        raise ValueError(f'Skill {name} has missing or inconsistent metadata')
    if fields['type'] not in ('task', 'style'):
        raise ValueError(f'Skill {name} has invalid type')
    instructions = parts[2].strip()
    if not instructions.startswith('# Skill:') and not instructions.startswith('# Style Skill:'):
        raise ValueError(f'Skill {name} is missing its instruction heading')
    return SkillDefinition(name=name, type=fields['type'],
                           description=fields['description'],
                           output_contract=fields['output_contract'],
                           version=int(fields['version']), instructions=instructions)


@lru_cache(maxsize=16)
def _instruction(name: str) -> str:
    return get_skill_definition(name).instructions


def skill_instructions(plan: SkillPlan) -> str:
    return '\n\n'.join(_instruction(name) for name in plan.names)
