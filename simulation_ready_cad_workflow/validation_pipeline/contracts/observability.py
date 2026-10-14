"""Published observability schema identifiers.

Consumers such as the dashboard read Loop state and event documents by
these schema IDs.  Keeping the identifiers here means readers never have
to import Loop execution code to recognize its published documents.
"""

from __future__ import annotations


LOOP_STATE_SCHEMA = "simulation_ready_cad_react_loop_state_v2"
LOOP_EVENT_SCHEMA = "simulation_ready_cad_react_loop_event_v2"
