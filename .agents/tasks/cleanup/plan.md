# Implementation Plan — quantum_traffic_optimizer cleanup, bug-fix & docs

Repo root (all paths below are relative to it):
`c:\Users\vtrsh\Downloads\Jetson Orin Traffic System Backup-20260919T085641Z-1-001\quantum_traffic_optimizer`

## Context discovered during exploration (facts, not assumptions)

Environment verified on this machine (Python 3.13.0rc2): all 18 `.py` modules pass
`python -m py_compile`. Every third-party dep (numpy, networkx, pandas, streamlit 1.62.0,
folium, streamlit-folium, cv2, ultralytics, torch, qiskit 2.5.2, twilio, dotenv, ncnn, plotly)
is installed, so import-level verification is possible here (do NOT assume a bare environment).

- `app.py` chain (`traffic_network`, `quantum_optimizer`, `classical_baseline`, `environmental`,
  `emergency`, `events`) imports and runs end-to-end: `optimize_signals` + `apply_assignment` +
  `evaluate_plan` + baselines all succeed.
- `quantum_optimizer.QISKIT_AVAILABLE` evaluates to **False** on this machine even though qiskit is
  installed, because the optional import block needs `qiskit_algorithms` (NOT installed) and
  `qiskit.primitives.Sampler` (removed in qiskit 2.x). The try/except correctly falls back to
  classical simulated annealing — this is the documented, working behavior, not a crash.
- `git init` done on branch `main` with **zero commits** and nothing staged. `.env`, `wheels/`,
  `traffic_log.json`, `traffic_logs.csv`, `accident_snapshots/`, `speed_violations/`, `__pycache__/`
  are already correctly ignored (they do not appear in `git status`). `redlight_violations/` is NOT
  ignored and shows as untracked.
- `redlight_violations/` (hundreds of JPGs) is a leftover from the OLD Jetson project — **no module in
  this repo references "redlight"** (verified by grep). Safe to delete from the working tree and ignore.
- `chatbot` / `openai` / `featherless` appear **nowhere** in the code (grep returned zero matches). The
  README's "Traffic chatbot (Featherless AI)" section and `chatbot.py` references are stale.
- `pages/` is **empty**. README does not actually reference it in prose, but it is an unused dir to remove.
- `live_intersection.py` is a valid, compiling helper that **nothing imports** (only the README-style
  diagram references the junction idea). Keep it (documented helper, harmless) but do not advertise it as wired in.
- `alerts.status_text()` and `alerts.twilio_available()` are defined but unused by `live_app.py`.
- `accident_ncnn_model/model_ncnn.py` is a Colab test stub with hard-coded `/content/...` paths and a
  `test_inference()` main; nothing imports it. It ships inside the model folder as a sample.
- Streamlit container-width API is inconsistent: `app.py` uses the older `use_container_width=True`;
  `live_app.py` uses the newer `width="stretch"`. Both work in Streamlit 1.62 — cosmetic only.
- Sibling folder `..\Jetson Orin Traffic System Backup\` and parent-level `*.mp4` files exist and MUST
  NOT be touched. `config.py` only READS a parent-level `car accident video.mp4` path as a fallback.

### Bug/defect inventory (file:line → fix)

1. **requirements.txt** — quantum extras are unusable as pinned. `qiskit>=1.0.0` with the code in
   `quantum_optimizer.py:46-54` requires `qiskit-algorithms` (missing) and `qiskit.primitives.Sampler`
   (gone in qiskit 2.x). Fix: add `qiskit-algorithms>=0.3.0` to the optional quantum block and add a
   comment pinning a known-good stack (qiskit 1.x line) so QAOA can actually load; keep the graceful
   annealing fallback untouched. This is the only functional defect; everything else runs. (No code change
   to the optimizer — preserve working fallback.)
2. **README.md** — documents a non-existent `chatbot.py` / Featherless AI feature and a `FEATHERLESS_API_KEY`
   / `openai` dependency. Fix: delete that whole section (and the `pages/` mention if present).
3. **.gitignore** — missing `redlight_violations/` and loose test videos `*.mp4`. Fix: add them.
4. **Dead/vestigial files** — `pages/` (empty), `redlight_violations/` (old-project output),
   generated `traffic_log.json` / `traffic_logs.csv` / `accident_snapshots/*` / `speed_violations/*` in the
   working tree. Fix: remove from the working tree (they stay gitignored and regenerate at runtime).

No undefined-name, broken-signature, or missing-import defects were found in the modules themselves.

### Verification commands (use these, not grep)
- Compile all: `python -m py_compile *.py` (PowerShell: `Get-ChildItem -Filter *.py | % { python -m py_compile $_.Name }`)
- App logic chain + run: `python -c "from traffic_network import TrafficNetwork; from quantum_optimizer import optimize_signals, apply_assignment; from classical_baseline import solve_fixed, solve_rule_based; from environmental import evaluate_plan, improvement; net=TrafficNetwork(6,7); net.randomize_traffic(0.5); r=optimize_signals(net); apply_assignment(net,r); m=evaluate_plan(net,r); f=solve_fixed(net); print(r.method, improvement(evaluate_plan(net,f), m))"`
- Live chain import (heavy deps present here): `python -c "import config, logger, alerts, signal_controller, savings_predictor; from video_detector import MultiVideoDetector, DIRECTIONS; from signal_controller import AdaptiveSignalController; from savings_predictor import SavingsPredictor; print('live chain OK')"`
- Streamlit smoke (no run needed): the two `st.set_page_config` apps only need import-safe modules; do not launch a server in CI.

---

## Ordered implementation items

- [ ] 1. Fix `requirements.txt` quantum + optional-dep hygiene.
      Add `qiskit-algorithms>=0.3.0` under the quantum block; add a comment that the QAOA path needs the
      qiskit 1.x line (qiskit, qiskit-aer, qiskit-optimization, qiskit-algorithms) because qiskit 2.x removed
      `qiskit.primitives.Sampler`; keep the "optional — classical annealing fallback" note. Group and label
      heavy/optional deps (torch/ultralytics, opencv, ncnn, twilio, qiskit*) clearly. Do NOT remove any dep
      the code imports; there is no `openai`/chatbot dep to drop (never present).
      Files: `requirements.txt`
      Verify: `python -c "import ultralytics, cv2, twilio, dotenv, networkx, folium, streamlit_folium, plotly; print('core deps import')"` and visually confirm the quantum block lists qiskit-algorithms.

- [ ] 2. Rewrite `README.md` to match the actual code.
      Remove the entire "Traffic chatbot (Featherless AI)" section and any `chatbot.py` / `FEATHERLESS_API_KEY`
      / `openai` / `pages/` references. Keep and accurately describe the real features: two Streamlit apps
      (`streamlit run app.py` network simulator, `streamlit run live_app.py` live YOLO), QUBO/QAOA optimizer
      with classical-annealing fallback, adaptive signals (`signal_controller.py`), emergency corridor,
      accident detection + optional Twilio SMS, speed violations, JSON/CSV logging. Update the "Project layout"
      table to list the files that actually exist (add `live_app.py`, `config.py`, `logger.py`, `alerts.py`,
      `model_loader.py`, `video_detector.py`, `signal_controller.py`, `simple_tracker.py`, `speed_tracker.py`,
      `savings_predictor.py`, `live_intersection.py`, `accident_ncnn_model/`). State plainly that the QAOA path
      needs the qiskit 1.x stack and otherwise falls back to annealing. Keep the Windows PowerShell + venv setup.
      Add a short "Secrets" note: copy `.env.example` → `.env`, never commit `.env`.
      Files: `README.md`
      Verify: `Select-String -Path README.md -Pattern "chatbot|Featherless|FEATHERLESS|openai|pages/"` returns no matches; `streamlit run app.py` / `streamlit run live_app.py` instructions still present.

- [ ] 3. Finalize `.gitignore`.
      Append `redlight_violations/` and `*.mp4` (loose test videos) and any other runtime output (e.g.
      `*.engine` already present). Keep `.env` ignored and `.env.example` committed. Do not remove existing rules.
      Files: `.gitignore`
      Verify: `git check-ignore redlight_violations/ ; git check-ignore .env` both report ignored; `git check-ignore .env.example` reports nothing (not ignored).

- [ ] 4. Tidy files/folders in the working tree (destructive — scoped to this repo only).
      Remove `pages/` (empty), `redlight_violations/` (old-project output, nothing writes it), and the
      regenerable generated artifacts `traffic_log.json`, `traffic_logs.csv`, `accident_snapshots/`,
      `speed_violations/` from the working tree. They remain gitignored and regenerate at runtime. Keep model
      weights (`yolo11s.pt` ~19MB, `accident_ncnn_model/*` ~44MB bin — each under 50MB). Keep `wheels/`
      on disk (gitignored) — do NOT commit. Do NOT delete `.env`. Do NOT touch anything outside this repo.
      Files: delete `pages/`, `redlight_violations/`, `traffic_log.json`, `traffic_logs.csv`,
      `accident_snapshots/`, `speed_violations/`
      Verify: `Test-Path pages, redlight_violations, traffic_log.json, traffic_logs.csv` all return False;
      `Test-Path yolo11s.pt, accident_ncnn_model\model.ncnn.bin, .env` all return True.

- [ ] 5. Re-verify the whole project still builds and runs after cleanup.
      Files: none (verification only)
      Verify: `Get-ChildItem -Filter *.py | % { python -m py_compile $_.Name }` exits clean for all modules;
      run the app-logic-chain one-liner from the Context section and confirm it prints the annealing method and
      an `improvement` dict; run the live-chain import one-liner and confirm `live chain OK`.

- [ ] 6. Create logical git commit(s) on `main` (NO remote, NO push).
      Stage the intended source set explicitly (do not `git add .` blindly — rely on `.gitignore` to keep
      `.env`, `wheels/`, generated data out, and confirm with `git status` first). Suggested single clean
      initial commit: all `.py` modules, `README.md`, `requirements.txt`, `.gitignore`, `.env.example`,
      `yolo11s.pt`, `accident_ncnn_model/` (weights + param + metadata). Commit message e.g.
      `chore: clean up repo, fix docs/deps, prepare for GitHub`.
      Before committing, confirm `.env` is not staged: `git status --short` must not list `.env`.
      Files: git index only
      Verify: `git log --oneline` shows the new commit; `git ls-files .env` prints nothing; `git ls-files`
      lists the source files and model weights but none of the generated/ignored artifacts.

## Notes / assumptions
- Optional consistency nicety (NOT required, skip unless cheap): align Streamlit container-width API —
  `app.py` uses `use_container_width=True`, `live_app.py` uses `width="stretch"`. Both work in Streamlit
  1.62; leaving them is fine and lower-risk than touching working UI code.
- `accident_ncnn_model/model_ncnn.py` and `live_intersection.py` are left in place: both are harmless and
  referenced by the project's story; removing them is out of scope for a cleanup-not-rewrite task.
- Committing `yolo11s.pt` and `accident_ncnn_model/model.ncnn.bin` to git is per the task (each < 50MB). If
  the user later wants a lean repo, Git LFS is the follow-up, but that is not requested here.
