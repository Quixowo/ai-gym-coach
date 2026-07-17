# Memory-extraction fixtures (`claude_responses/mem_*.json`)

These drive the 4th eval suite, `tests/test_memory_extraction.py`, in the same
recorded-fixture style as the Phase 7 suites (`tc_*`, `rf_*`, `grd_*`): committed JSON
holds the model reply, the real production code runs against a fake client replaying
it, and CI never calls a live API.

## Status

Each file carries a `"hand_authored"` flag (presence asserted by
`test_all_fixtures_declare_recording_status`): `true` = hand-authored, shaped exactly
like a real recorded Haiku reply; `false` = re-recorded from a live Haiku call by the
memory phase gate. The adversarial fixtures (`mem_04`–`mem_07`: malformed/unparseable
model output a live model wouldn't reproduce) stay hand-authored permanently by design.

## Fixture shape

| field | meaning |
|---|---|
| `id` | fixture id (also the filename stem) |
| `hand_authored` | `true` until a live recording replaces `recorded_reply` |
| `user_message` / `assistant_reply` | the one turn extraction runs over |
| `existing_keys` (optional) | topic_keys the test pre-seeds as prior observations |
| `recorded_reply` | **the raw model text** — `resp.content[0].text` from the extraction `messages.create` call (a JSON array, possibly wrapped in prose/code fences) |
| `expected` | the validated observations, `== _parse_observations(recorded_reply)` |

`recorded_reply` is the exact field a live recorder writes (mirrors `recorded_verdict`
in `rf_*` and `synthesis_text` in `grd_*`). `expected` is **derived**, not
independently authored: it is `_parse_observations(recorded_reply)`, so it already
reflects the code's drop/normalize/truncate rules.

## When the live memory gate runs

For each fixture, call Haiku's extraction prompt once, write the real
`resp.content[0].text` into `recorded_reply`, then recompute `expected` by running
`app.services.memory_service._parse_observations` over it. Flip `hand_authored` to
`false`. `test_expected_matches_parser` guards that `expected` stays in sync with
`recorded_reply`, so a mismatch after re-recording fails loudly.
