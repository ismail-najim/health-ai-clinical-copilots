# clinical-copilots

Five AI tools that help a clinician with everyday work. All of them are assistants — a person always stays in charge, and each one runs offline with a built-in stand-in so you can try it with no setup.

| Folder | What it does | Runs offline, no key? |
|---|---|---|
| **[scribe](scribe/)** | Turns a visit conversation into a written note, and links every line back to what was actually said — flagging anything it can't back up | Yes |
| **[guideline-cds](guideline-cds/)** | Answers medical questions using only real guideline text, cites every claim, and refuses to answer if it can't find support | Yes |
| **[ehr-agent](ehr-agent/)** | Reads a patient record and drafts orders — but never writes anything without a human's approval | Yes |
| **[documentation](documentation/)** | Writes pre-visit briefs, discharge summaries, and plain-English rewrites, with each line grounded or flagged | Yes |
| **[message-triage](message-triage/)** | Sorts incoming patient messages by urgency and drafts a reply — checking for emergencies first, and leaving every send to a clinician | Yes |

Each folder is its own self-contained project with its own README and steps. They don't share code — pick one, open its folder, and follow the steps there.

## First thing to do in any folder
Each tool runs fully offline with a built-in stand-in for the AI model — no API key and no internet needed just to try it:

```bash
cd scribe        # or guideline-cds, ehr-agent, documentation, message-triage
pip install -r requirements.txt
python -m pytest tests/ -q
```

If that passes, you're set. To get real answers instead of the stand-in, set an Anthropic API key — each folder's README explains how.

## Important
These are **assistants that help a clinician who stays in charge — not autonomous, and not medical devices.** Each folder's `model_card.md` has the details. The code is MIT-licensed (see `LICENSE`).
