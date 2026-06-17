"""Prometheus V9PRO Ontology Generator — Dynamic entity/relation discovery.

V8 had core/ontology_generator.py (199 lines). V9PRO adds:
runtime registration of new EntityType/RelationType, and LLM-powered
concept discovery from text content.

This is the data-layer implementation of T1 (涌现>设计):
the system discovers new concepts at runtime instead of relying on
static enum definitions.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from prometheus_v10.schema import NodeType

logger = logging.getLogger(__name__)


@dataclass
class EntityPattern:
    """A discovered entity type pattern."""
    name: str = ""
    examples: list[str] = field(default_factory=list)
    frequency: int = 0
    confidence: float = 0.0
    source: str = "heuristic"  # heuristic | llm | manual


@dataclass
class RelationPattern:
    """A discovered relation type pattern."""
    name: str = ""
    subject_type: str = ""
    object_type: str = ""
    examples: list[str] = field(default_factory=list)
    frequency: int = 0
    confidence: float = 0.0


class OntologyGenerator:
    """Dynamic ontology: discover and register new entity/relation types.

    V8's schema.py had static NodeType enum. V9PRO allows the system
    to discover new concepts at runtime and register them as typed
    patterns. This is T1 (涌现>设计) at the data layer.

    Discovery methods:
    1. Heuristic: pattern matching from content (capitalized terms,
       quoted phrases, technical terms)
    2. LLM: extract structured concepts from text
    3. Manual: explicit registration by user/agent
    """
    def __init__(self) -> None:
        self._entities: dict[str, EntityPattern] = {}
        self._relations: dict[str, RelationPattern] = {}
        self._discovery_count = 0
        self._llm = None

        # Initialize with built-in types from schema
        for nt in NodeType:
            self._entities[nt.value] = EntityPattern(
                name=nt.value, frequency=0, confidence=1.0, source="builtin",
            )

    def set_llm(self, llm: Any) -> None:
        """Set LLM for concept discovery."""
        self._llm = llm

    def register_entity(self, name: str, examples: list[str] | None = None,
                        source: str = "manual") -> EntityPattern:
        """Register a new entity type."""
        name_lower = name.lower().replace(" ", "_")
        if name_lower in self._entities:
            # Update existing
            existing = self._entities[name_lower]
            if examples:
                existing.examples.extend(examples)
            existing.frequency += 1
            existing.confidence = min(1.0, existing.confidence + 0.1)
            return existing

        pattern = EntityPattern(
            name=name_lower,
            examples=examples or [],
            frequency=1,
            confidence=0.5 if source == "heuristic" else 0.8,
            source=source,
        )
        self._entities[name_lower] = pattern
        logger.info(f"Registered entity type: {name_lower} (source={source})")
        return pattern

    def register_relation(self, name: str, subject_type: str, object_type: str,
                          examples: list[str] | None = None) -> RelationPattern:
        """Register a new relation type."""
        key = f"{subject_type}_{name}_{object_type}"
        if key in self._relations:
            existing = self._relations[key]
            existing.frequency += 1
            return existing

        pattern = RelationPattern(
            name=name, subject_type=subject_type, object_type=object_type,
            examples=examples or [], frequency=1, confidence=0.7,
        )
        self._relations[key] = pattern
        logger.info(f"Registered relation: {subject_type} --{name}--> {object_type}")
        return pattern

    def discover_from_text(self, text: str) -> list[EntityPattern]:
        """Discover entity patterns from text content.

        Heuristic method: looks for capitalized terms, quoted phrases,
        technical patterns (X-Y, X.Y, X::Y).
        """
        self._discovery_count += 1
        discovered: list[EntityPattern] = []

        # Pattern 1: Capitalized terms (potential proper nouns/concepts)
        caps_pattern = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', text)  # CamelCase
        for term in set(caps_pattern):
            if term.lower() not in self._entities:
                pattern = self.register_entity(term, [term], source="heuristic")
                discovered.append(pattern)

        # Pattern 2: Quoted phrases
        quoted = re.findall(r'"([^"]{3,30})"', text)
        for phrase in set(quoted):
            name = phrase.lower().replace(" ", "_")
            if name not in self._entities and len(name) > 3:
                pattern = self.register_entity(name, [phrase], source="heuristic")
                discovered.append(pattern)

        # Pattern 3: Technical patterns (X-Y, X.Y)
        tech = re.findall(r'\b[A-Za-z]+[-.][A-Za-z]+[-.A-Za-z]*\b', text)
        for term in set(tech):
            name = term.lower().replace("-", "_").replace(".", "_")
            if name not in self._entities and len(name) > 3:
                pattern = self.register_entity(name, [term], source="heuristic")
                discovered.append(pattern)

        return discovered

    def discover_with_llm(self, text: str) -> list[EntityPattern]:
        """Use LLM to discover concepts from text."""
        if not self._llm:
            return self.discover_from_text(text)

        try:
            import json
            prompt = (
                "Extract the key concepts and entity types from the following text. "
                "Return a JSON array of objects with 'name' and 'examples' fields. "
                "Each name should be a lowercase_underscore type name. "
                "Return ONLY the JSON array.\n\n"
                f"Text:\n{text[:2000]}"
            )
            response = self._llm(prompt).strip()
            if response.startswith("```"):
                response = response.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            concepts = json.loads(response)
            discovered = []
            if isinstance(concepts, list):
                for item in concepts[:10]:
                    name = item.get("name", "")
                    examples = item.get("examples", [])
                    if name and len(name) > 2:
                        pattern = self.register_entity(name, examples, source="llm")
                        discovered.append(pattern)
            return discovered
        except Exception as e:
            logger.warning(f"LLM discovery failed: {e}")
            return self.discover_from_text(text)

    def get_entity(self, name: str) -> EntityPattern | None:
        return self._entities.get(name.lower())

    def get_relation(self, name: str) -> list[RelationPattern]:
        return [r for r in self._relations.values() if r.name == name]

    def get_all_entities(self) -> list[EntityPattern]:
        return list(self._entities.values())

    def get_all_relations(self) -> list[RelationPattern]:
        return list(self._relations.values())

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "entity_count": len(self._entities),
            "relation_count": len(self._relations),
            "discovery_count": self._discovery_count,
            "by_source": defaultdict(int, {
                src: sum(1 for e in self._entities.values() if e.source == src)
                for src in set(e.source for e in self._entities.values())
            }),
        }
