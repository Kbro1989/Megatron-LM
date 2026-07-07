# King Wen Megatron Curriculum
**Scope:** Jarvis-native training only. No external priors. No semantic mappings.
**Source of truth:** `C:\Users\krist\Desktop\KING-WEN-I-CHING-IMMUTABLE-TABLES\kingwen_ternary_tables_complete.py`

## Slot grammar
```
[KING_WEN_ORACLE]
hexagram=<id> <name> | category=<category> | action=<action>
trigrams=<upper> over <lower> | binary=<binary>
phase=<phase_bits> <phase_temporal> | polarity=<phase_polarity> | <phase_description>
inject_site primary=<primary_pool> secondary=<secondary_pool> porosity=<porosity> <porosity_label>
inject_reason=<reason>
expanded_vector voiceWeight=<vw>, coherence=<co>, chaos=<ch>, whimsy=<wh>, darkTone=<dt>
resolved_vector voiceWeight=<vw>, coherence=<co>, chaos=<ch>, whimsy=<wh>, darkTone=<dt>
line_states=<yao checklist summary>
[END_ORACLE]
```

```
[SOVEREIGN_PIPELINE_SCENE]
Scene hexagram=<id> <name> | category=<category> | action=<action>
Temporal phase=<phase_bits> <phase_temporal> | status=<phase_description>
Delivery primary=<primary_pool> | secondary=<secondary_pool> | porosity=<porosity>
Inject reason=<reason>
Primary vector voiceWeight=<vw>, coherence=<co>, chaos=<ch>, whimsy=<wh>, darkTone=<dt>
Secondary vector ...
Porous mix vector ...
Base expanded ...
Resolved expanded ...
Oracle checks=<checklist status lines>
Scene boundary=<binary> | upper=<upper_trigram> | lower=<lower_trigram>
[END_SCENE]
```

## Label schema
Required fields, sourced only from `collapse_full_128()`:
- `hexagram_id`: int
- `phase_bits`: int 0-7
- `phase_temporal`: past|present|future
- `phase_polarity`: stable_yin|stable_yao|stable_yang
- `inject_site`: primary_pool, secondary_pool, porosity, reason
- `expanded_vector`: 5-axis continuous
- `resolved_vector`: 5-axis continuous
- `checklist`: 6 axis directions + status + value
- `source`: `collapse_full_128`

## Loss weighting
- Structural fields (`hexagram_id`, `phase_bits`, `inject_site`): 1.0
- 5-axis vector fields (`voiceWeight`, `coherence`, `chaos`, `whimsy`, `darkTone`): 1.2
- Checklist value fields: 0.8
- Domain/source tokens: 0.5

## Constraints
- No hardcoded emotional vectors anywhere in this file.
- No lookup tables.
- No semantic mappings.
- All values must trace back to immutable tables or `collapse_full_128()` output.
