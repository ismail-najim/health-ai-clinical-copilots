# message-triage — sort patient messages and draft replies (a clinician always approves)

It checks every incoming patient message for an emergency first, sorts what's
left by urgency, condenses it to the real question, and drafts a reply. A
clinician approves every send; nothing goes to a patient on its own.

## What you need

- Python 3.11
- No GPU
- No API key to try it: a deterministic offline stand-in is built in, so the
  whole tool runs on your CPU. Set `ANTHROPIC_API_KEY` when you want real drafts.

## Easiest way to see it work

```bash
pip install -r requirements.txt
python -m pytest tests/ -q
```

The tests run offline with no key and no downloads. You should see a line like
`6 passed`.

## Run it

```bash
python -m app.demo
```

Paste a patient message and, optionally, a few record facts (one per line). You
get an urgency label, the condensed question, and a draft reply — or an
emergency banner. To send, a clinician types their name and clicks Approve &
send.

For real drafts instead of the offline stand-in, set your key first:

```bash
export ANTHROPIC_API_KEY=sk-...
python -m app.demo
```

## What you'll see

- **An urgency label** — emergency, urgent, or routine.
- **The condensed question** — the one thing the patient is actually asking,
  pulled out of the surrounding text.
- **A draft reply** — warm, plain, grounded in any record facts you gave it, and
  marked "awaiting clinician approval". Or, if the message looks like an
  emergency, an emergency banner instead of a draft.

The emergency-first rule is simple: before any model runs, a fixed pattern check
looks for the signs of an emergency — chest pain, trouble breathing, stroke
signs, suicidal thoughts, severe bleeding, and so on. If it finds one, you get
"seek emergency care" guidance and the message is routed to the on-call
clinician. It never gets a routine reply.

## If something doesn't work

- `No module named src` — run the commands from the project folder, not a
  subfolder.
- The tests want an API key — they shouldn't. They use the offline stand-in;
  make sure you ran `python -m pytest tests/ -q` and not the demo.
- The demo won't open — check that `gradio` installed cleanly with
  `pip install -r requirements.txt`.
- You want real drafts and they look canned — set `ANTHROPIC_API_KEY`; without
  it the tool uses the deterministic offline stand-in on purpose.

## Safety

- Emergencies are caught by a fixed rule that the AI can't override. It runs
  before any model and sits above it, so no prompt or model output can switch it
  off.
- A clinician approves every reply. There is no path that sends on its own.
- This is not a medical device. It does not diagnose or order medications; it
  helps a care team triage and draft.
- Missing an emergency is the worst error this tool can make, so the gate is
  tuned to over-warn. It will raise some false alarms on purpose — a false alarm
  is safe, a missed emergency is not.

## Learn more

- `ARCHITECTURE.md` — the pipeline and every file, in plain terms.
- `model_card.md` — intended use, data, metrics, and limitations.
- `config.yaml` — model name, urgency classes, and metric targets.
