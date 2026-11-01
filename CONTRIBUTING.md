# Contributing to Reelferry

Reelferry is open source under the Apache License 2.0 and is based on BridgeClip (MIT).
Bug reports, feature requests and pull requests are welcome.

## Set up

Follow the [development instructions](README.md#develop).
Install the in-repo engine dependencies in `engine/.venv`, then run `npm ci` and `npm run dev`.
Keep provider keys in the app's Settings; tests do not need real keys.

## Make a change

- Keep Electron main-process authority, preload IPC, renderer UI, and Python bridge responsibilities separate.
  Validate data at every IPC, subprocess, and saved-file boundary.
- Keep provider keys out of renderer state, logs, test fixtures, and issue reports.
  Use placeholders in examples.
- Update the README or architecture guide when setup, provider data flow, or supported behavior changes.
- Add a regression test for a bug or a new boundary.
  Avoid tests that only repeat implementation details.
- Keep license notices intact: `LICENSE`, `licenses/BridgeClip-MIT.txt`, `engine/LICENSE` and the third-party notices.

Run `npm run typecheck`, `npm run lint`, `npm test`, and `npm run build`.
For engine changes, install `pytest` in `engine/.venv` (`engine/.venv/bin/pip install --require-hashes -r scripts/python-tools/pytest.lock`) and run `engine/.venv/bin/python -m pytest -q engine/tests`; the runtime lockfile does not include test tools.
Describe behavior, tests, and any user-visible screenshots or sample outputs in the PR.

Use an imperative, scoped commit message such as `fix(clips): validate saved run output`.
By contributing, you agree to follow the [code of conduct](CODE_OF_CONDUCT.md).
