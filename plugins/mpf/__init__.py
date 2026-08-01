"""MPF (Download and Upload Form) plugin for ATLAS AI.

Makes the agent behave like a human data-entry operator against the MPF
desktop form: reads the LEFT source panel, fills the RIGHT form, clicks
"Upload Details", waits for the next record and repeats until stopped.

The plugin never uses fixed screen coordinates - everything is discovered
semantically (window title, panel geometry, label aliases) each cycle.
"""

__all__ = ["MpfPlugin"]
