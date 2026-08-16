# Model card: message-triage

## What it is

An assistant for a patient-portal inbox. It sorts an incoming patient message by
urgency, condenses it to the question the patient is really asking, and drafts a
reply for a clinician to review. It checks for emergencies before anything else,
and it never sends a reply on its own.

## Intended use

A support tool for a care team working through portal messages. It helps triage
and draft; a clinician remains the sender of record and makes every decision.

## Out of scope

- It is not a medical device and does not diagnose, order medications, or change
  treatment.
- It is not patient-facing. Patients never interact with it directly, and no
  reply reaches a patient without a clinician approving it.
- It does not auto-reply, auto-close, or triage without a human in the loop on
  the send.

## How it works

1. A deterministic emergency gate (src/redflags.py) scans the raw message for
   emergency signs. It is a fixed rule, not a model, and sits above everything
   else. On a hit, the pipeline returns emergency guidance and stops.
2. An urgency classifier (src/triage.py) labels the rest emergency / urgent /
   routine. A small scikit-learn baseline and a model-based option are both
   available and meant to be compared.
3. A summarizer (src/summarize.py) condenses the message to one question, with a
   check against inventing details the message never stated.
4. A drafter (src/reply.py) writes a warm, grounded reply that never reassures a
   patient out of seeking care. The draft is held for approval.
5. A clinician approves; that is the only path to a sent reply.

## Data

- MeQSum (open, CC-BY): condense the message to its real question.
- PMR-Bench (gated, accept terms): message urgency labels for training the
  baseline on real messages.
- MedRedQA (registration and data-use agreement, research only): studied for
  reply structure during development; not embedded in a shipped model.
- MedRedFlag (check terms): probe for the "correct a misconception, don't affirm
  it" behaviour.

## Metrics

- **Red-flag recall (headline): target ~1.0.** The fraction of true emergencies
  the gate catches. Reported on its own. A single missed emergency is a failure.
- **Over-reassurance rate (tracked separately): target ~0.** The fraction of
  drafts that soothe a patient without pointing them back to care.
- Triage accuracy and summary faithfulness are reported alongside, never blended
  into the safety number.

## Safety and limitations

- Missing an emergency is the worst error this tool can make, so the gate is
  tuned to over-warn. It will raise some false alarms, and that is by design.
- The urgency classifier can be wrong; it is a routing aid, not a diagnosis, and
  the emergency gate — not the classifier — is the safety backstop.
- Drafts can be imperfect or incomplete. They exist to save a clinician time,
  not to replace clinical judgment. Every send is a human decision.
- When a real API key is set, de-identify messages before they leave your
  environment, in line with your own privacy obligations.
