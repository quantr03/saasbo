# Makefile for docs/thesis/ -- the synthobj thesis appendix.
#
# tables and check invoke regen_tables.py, which Task 2 has not created yet;
# both targets will fail until then. pdf and lint are this task's deliverable.

PYTHON ?= /opt/anaconda3/envs/saasbo/bin/python

.PHONY: tables check study-tables study-check pdf lint

tables:
	$(PYTHON) regen_tables.py

check:
	$(PYTHON) regen_tables.py --check

# The R2-D2 study's tables, from sagp_analysis/r2d2/latest (pass SNAPSHOT=<dir> to read another).
SNAPSHOT ?= ../../sagp_analysis/r2d2/latest

study-tables:
	$(PYTHON) regen_study_tables.py --snapshot $(SNAPSHOT)

study-check:
	$(PYTHON) regen_study_tables.py --check --snapshot $(SNAPSHOT)

pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error standalone_main.tex

lint:
	@if command -v chktex >/dev/null 2>&1; then \
		chktex -q -I0 appendix-synthobj.tex A1-construction.tex A2-families.tex A3-api-fileformat.tex A4-design-decisions.tex; \
	fi
	$(PYTHON) lint_tex.py
