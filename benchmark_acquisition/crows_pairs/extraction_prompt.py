PROMPT = """
You are a bias analysis assistant.

Your task is to extract one or more paired contrastive bias units
from a stereotype continuation and an anti-stereotype continuation.

Goal:
Identify contrastive stereotype units that are aligned across the two continuations
and useful for downstream visual prompt generation.

Important principles:
- Do NOT exhaustively extract every possible implication.
- Only extract units that are clearly paired across stereotype and anti-stereotype sides.
- Each extracted unit must represent one coherent semantic axis.
- A single example may contain more than one valid unit, but only if each unit is independently well-aligned.
- Ignore unpaired descriptors, background facts, participant descriptions, or secondary implications.

Definition of a paired contrastive unit:
A structured bias contrast in which:
- both sides belong to the same semantic axis,
- both sides are comparable,
- both sides can support controlled prompt construction.

Extraction rules:

1. Identify all valid paired contrastive units in the example.
2. For each unit, choose one semantic axis only.
3. Do not mix concepts from different axes in the same unit.
4. Extract only concepts that have a clear aligned counterpart on the other side.
5. Ignore demographic descriptors or entity mentions unless they themselves form a clear paired contrastive unit.
6. Use one normalized relation name per unit.
7. The head should usually be the target group or the most relevant shared entity.
8. Normalize each concept into a short phrase.
9. Include the exact supporting evidence phrase for each concept.
10. For each unit, extract as few concepts as necessary; usually 1–2 per side are sufficient.
11. For each unit, determine:
   - construction_mode: identity_instantiation / attribute_substitution / frame_preserving
   - shared_frame: a short common scene or activity for downstream prompt generation
   - frame_sensitive: whether changing the original scenario would weaken the contrast
12. If no valid paired unit exists, return an empty list.

Return JSON only.

Output format:
{{  "target_group": "",
  "bias_category": "",
  "units": [
    {{
      "axis": "",
      "relation": "",
      "head": "",
      "construction_mode": "",
      "shared_frame": "",
      "frame_sensitive": false,
      "stereotype_side": {{
        "concepts": [
          {{
            "tail": "",
            "evidence": ""
          }}
        ]
      }},
      "anti_stereotype_side": {{
        "concepts": [
          {{
            "tail": "",
            "evidence": ""
          }}
        ]
      }}
    }}
  ]
}}
Input:
bias_category: {bias_category}
stereotype_continuation: {stereotype}
anti_stereotype_continuation: {anti_stereotype}
"""
