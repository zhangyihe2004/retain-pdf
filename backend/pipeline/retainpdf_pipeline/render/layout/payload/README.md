# Payload Layout Policies

`body_pipeline.py` is the orchestration layer for body text layout. Keep it as a stage list only. It depends on `body_policy_facade`, not on every leaf helper.

Policy boundaries:

- `flow_text.py`: the shared ordinary-text/specialized-layout routing decision.
- `field_style.py`: source-derived style preferences for high-confidence repeated field rows. It never changes coordinates or decides whether translated text fits.
- All text reuses the original typography stages, with one shared font/field-style finalization. Ordinary flow still uses a single geometric height limit and the shared Typst glyph fitter as its final capacity authority; inherited typography never grants extra capacity.
- Verified horizontal expansion is a separate final-fit ceiling (`_render_available_width_pt`), never part of `inner_bbox`, typography, density, source lines or collision policy. Measured text retains its original width at the preferred typography when it fits, otherwise borrows only the minimum required width before the existing font/leading fit. Unmeasured special layouts keep their original width. Lists have explicit item spacing; measurement and output use the same style.
- Original width is derived from the unchanged block rectangle, not stored again in models or manifests. Source-verified capacity is calculated once during final page preparation; the earlier geometry/indent prewarm only collects original metrics. Title budgets remain vertical-only, and unused horizontal capacity never changes overlay fills.
- Capacity starts at the page edge with the existing safety gap, then respects neighboring blocks, inferred columns, and source-pixel barriers. The current block or the widest aligned text block is not itself a capacity ceiling; solitary text uses the same checks.
- `body_common.py`: shared body-context predicates and geometry helpers.
- `body_policy_facade.py`: stable stage API consumed by `body_pipeline.py`.
- `body_font_dense_policy.py`: dense-body pressure tightening, force-fit marking, and underfilled body growth.
- `body_font_inheritance_policy.py`: short body line font inheritance from same-column body anchors.
- `body_font_unify_policy.py`: same-column similar-font unification.
- `body_font_harmonize_policy.py`: long body paragraph font/leading harmonization.
- `body_font_policy.py`: stable facade for body font stages. Pipeline code imports this facade so leaf font files can move without changing orchestration.
- `body_leading_policy.py`: body leading and line-spacing decisions.
- `body_fit_policy.py`: Typst fit budget decisions, including relaxed height for short body boxes.
- `body_smoothing_policy.py`: adjacent body-pair smoothing.

New rendering rules should be placed in the narrowest policy module that owns the field being mutated, then exposed through `body_policy_facade.py` if the pipeline needs to run it. Avoid adding layout rules directly to `body_pipeline.py`, and avoid making the pipeline import leaf helpers one by one.
