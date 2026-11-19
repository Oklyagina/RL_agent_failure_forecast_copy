"""User-facing workflows behind ``main.py`` (demo, train, smoke, status).

Modules are imported lazily by ``main.py`` because ``smoke`` must set its
environment overrides before ``project_config`` is first imported.
"""
