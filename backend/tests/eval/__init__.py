"""Manual/QA evaluation utilities (spec sections 18-19).

Nothing in this package is named test_*.py or discovered by pytest -- it
exercises the real embedding model, the real FAISS index, and (optionally)
the real Gemini API, none of which belong in the offline CI test suite.
"""
