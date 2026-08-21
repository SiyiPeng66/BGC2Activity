PYTHON ?= python3
CONTRACT_TESTS = PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v

.PHONY: test test-full audit-data

# Dependency-free contract layer + release-data audit.
test:
	$(CONTRACT_TESTS)

# Full suite including model modules (requires torch, pandas, scikit-learn).
test-full:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v

audit-data:
	$(PYTHON) scripts/audit_release_data.py
