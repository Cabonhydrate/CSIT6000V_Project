"""Compatibility routing over ONE canonical file-backed Skill registry.

Historical callers expect select_skill(question, profile, enabled) -> (name, text).
Skill bodies now live only in app/skills/*/SKILL.md, not in this module.
"""
from ..skills.registry import SkillPlan, skill_instructions


def select_skill(question, profile, enabled=True):
    if not enabled:
        return None, ''
    text = question.casefold()
    if any(x in text for x in ('学习路径', '学习计划', '三阶段', '先修知识', 'prerequisite', 'study plan')) or ('保存' in text and '学习' in text and ('下一步' in text or '哪些知识' in text)):
        name = 'suggestLearningPath'
    elif any(x in text for x in ('比较', '对比', 'difference', 'compare', 'versus')):
        name = 'comparePapers'
    elif any(x in text for x in ('导读', '阅读这篇论文', '带我阅读', '带我读', '阅读论文', '论文导读', 'summarize paper', 'read this paper')):
        name = 'guidePaperReading'
    elif profile.get('explanation_level') == 'beginner' or any(
            x in text for x in ('解释', '为什么', 'explain', 'how does')):
        name = 'explainForBeginner'
    else:
        return None, ''
    return name, skill_instructions(SkillPlan(None, (name,)))
