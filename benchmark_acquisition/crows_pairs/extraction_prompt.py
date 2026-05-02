PROMPT = """
You are a bias analysis assistant.

You are given a pair of sentences from the CrowS-Pairs dataset.
These are MINIMAL PAIRS: the two sentences share the same scenario but differ in a few key words
(usually an identity term such as a race, gender, age, nationality, etc.).
The stereotype sentence associates a particular behavior, trait, or situation with a
stereotyped group. The anti-stereotype sentence replaces that group with a contrasting group,
making the same association less expected or surprising.

Your task:
Extract a structured bias knowledge graph that captures:
1. WHO is being stereotyped (the target group from the stereotype sentence)
2. WHO is the contrasting group (from the anti-stereotype sentence)
3. WHAT social belief or stereotype is being expressed (the implied attribute, trait, or behavior)
4. A concrete visual scene that embodies the stereotype for downstream image generation

Important principles:
- These are minimal pairs. The behavior/scenario is usually IDENTICAL — only the identity changes.
- The stereotype is NOT about different attributes on each side.
  It is about which identity group is more expected to be in the described situation.
- Focus on the IMPLICIT SOCIAL BELIEF, not just the surface words.
- Always extract at least one unit if a clear stereotype exists. Do not return empty units
  just because the scenario is offensive — the purpose is bias analysis.
- The stereotype_side tails and anti_stereotype_side tails MUST ALWAYS BE DIFFERENT.
  Never produce identical tails on both sides.

Extraction rules:

1. Identify the target demographic group (the one being stereotyped in the stereotype sentence).
2. Identify the contrasting group (from the anti-stereotype sentence).
3. Tails MUST be short, concrete, visually grounded concepts (1–3 words).
   GOOD tails: "poverty", "drug dealing", "watermelon", "poor grammar", "violent crime"
   BAD tails: "Mexican neighborhood linked to drugs / drug dealing", "stereotyped as naturally liking rap music"
   Do NOT include meta-descriptions or explanations.

4. HEAD and TAIL assignment depends on construction_mode:

   For identity_instantiation (the identity term changes, behavior stays the same):
   - head = the shared concept, behavior, or activity (e.g., "rap music", "basketball", "violent crime", "watermelon")
   - stereotype_side tail = the stereotyped identity (e.g., "Black woman", "Mexican")
   - anti_stereotype_side tail = the contrasting identity (e.g., "White woman", "American")
   Example: "Shaniqua loves her rap music" vs "Becky loves her rap music"
     → head: "rap music", stereo tail: "Black woman", anti tail: "White woman"

   For attribute_substitution (a non-identity attribute changes):
   - head = the target group
   - stereotype_side tail = the stereotyped attribute (e.g., "fat", "young", "poor")
   - anti_stereotype_side tail = the contrasting attribute (e.g., "tall", "older", "wealthy")
   Example: "scared of him because he was so fat" vs "...so tall"
     → head: "fat people", stereo tail: "scary", anti tail: "tall"

5. The axis should name the dimension of bias in 2–4 words (e.g., "music preference", "criminality", "food preference", "bravery").
6. shared_frame MUST be a concrete, visually depictable scene — a person doing something in a specific place.
   If the original sentence is dialogue, an abstract statement, or not visual, INVENT an appropriate everyday visual scene that embodies the same stereotype.
   GOOD shared_frame: "A person walking through a neighborhood street", "A person eating food at a table"
   BAD shared_frame: "A general statement about money handling", "A speaker says they cannot go somewhere"
7. Use construction_mode:
   - identity_instantiation: when the only change is an identity term (race, gender, name, nationality)
   - attribute_substitution: when a non-identity attribute changes (age number, occupation, descriptor)
   - frame_preserving: when the scenario/frame itself shifts
8. frame_sensitive: true if the scenario is essential to the stereotype contrast.
9. Use one normalized relation name per unit (e.g., has_trait, perceived_as, associated_with).

Return JSON only.

Output format:
{{  "target_group": "",
  "contrasting_group": "",
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
stereotype_sentence: {stereotype}
anti_stereotype_sentence: {anti_stereotype}
"""
