"""Composable answer skills for the local paper tutor.

Skills describe *how to answer*. Retrieval and source verification remain in Agent.
"""
from .registry import SkillPlan, select_answer_skills, skill_instructions
from .outputs import normalize_skill_output, validate_skill_output

__all__ = ['SkillPlan', 'select_answer_skills', 'skill_instructions',
           'normalize_skill_output', 'validate_skill_output']
