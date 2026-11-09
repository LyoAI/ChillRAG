"""Multi-turn tool discipline: the history reminder and message composition.

On turn 2+ an episode's transcript holds only final answers — never the tool
rounds that produced them — so the strongest pattern in context is "answer
directly", and the model answers (possibly from memory) without searching.
These tests pin the counterweight: a retrieval reminder appended beside the
question only when history is present, in the episode's prompt language.
"""

from __future__ import annotations

import pytest

from chillrag.agent import ReActAgent
from chillrag.agent.prompts import (
    build_user_prompt,
    get_retrieval_reminder,
    prompt_lang,
)
from chillrag.tools import ToolRegistry
from chillrag.tools.searcher import DocumentStore


def _agent(**kwargs) -> ReActAgent:
    """An agent with no LLM over an empty tool set — message shaping needs neither."""
    return ReActAgent(None, ToolRegistry({}, DocumentStore(name="t")), **kwargs)


def test_first_turn_asks_the_question_bare():
    """No history, no reminder: the first turn behaves exactly as before."""
    assert _agent()._user_message("怎么开机", has_history=False) == {
        "role": "user",
        "content": "怎么开机",
    }


def test_later_turns_carry_the_reminder_beside_the_question():
    agent = _agent()
    content = agent._user_message("那功耗呢", has_history=True)["content"]
    assert content.startswith("那功耗呢")
    assert get_retrieval_reminder(agent.lang) in content


def test_reminder_joins_a_caller_shaped_user_message():
    """A caller's output contract and the reminder coexist in one message."""
    contract = "Answer in JSON."
    agent = _agent(user_prompt=lambda q: build_user_prompt(q, contract), lang="en")
    content = agent._user_message("list sections", has_history=True)["content"]
    assert contract in content
    assert get_retrieval_reminder("en") in content


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_reminder_exists_in_every_prompt_language(lang):
    assert get_retrieval_reminder(lang).strip()
    assert get_retrieval_reminder(lang) == get_retrieval_reminder(prompt_lang(lang))


def test_unknown_language_is_rejected_at_construction():
    with pytest.raises(ValueError, match="language"):
        _agent(lang="fr")


def test_build_user_prompt_composition():
    """The question travels verbatim; an output requirement rides along."""
    assert build_user_prompt("Q") == "Q"
    assert build_user_prompt("Q", "one line").endswith("one line")
