"""ehr-agent: read a patient record, draft an order, let a human approve it.

The package is small on purpose:

- ``llm``   one tiny model interface with a real client and an offline stand-in.
- ``fhir``  an in-memory fake patient store (and notes on a real FHIR server).
- ``tools`` the read-only tools the agent uses, plus a draft-only propose tool.
- ``agent`` a bounded loop: read the chart, draft an order, return a proposal.
- ``eval``  checks that nothing is auto-written and every draft is justified.

Nothing here ever writes to the record on its own. The agent only proposes;
a person approves or rejects.
"""

__version__ = "0.1.0"
