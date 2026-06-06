ROOT          := $(CURDIR)
SCRIPTS_DIR   := $(ROOT)/scripts
REPORT_SCRIPT := $(SCRIPTS_DIR)/export_report.py
MODELS_MD     := $(ROOT)/models.md
OPS_YAML      := $(ROOT)/ops.yaml
OPS_MD        := $(ROOT)/ops.md
EXCLUSIONS    := $(ROOT)/export-exclusions.yaml

# Per-model subprocess budget and the dynamic-shape sub-check budget. Both are wall-clock,
# so both have to move together: a dynamic budget above the subprocess one would just get
# the whole worker killed, losing that model's params, FLOPs and operators too.
TIMEOUT          ?= 120
DYNAMIC_TIMEOUT  ?= 60

.PHONY: report report.ci report.exclusions report.dry-run check-tree-clean

# ── timm export report ───────────────────────────────────────────────────────

# Regenerate models.md (torch.export compatibility, weight size, FLOPs) plus the aten op
# cross-reference ops.yaml/ops.md, all from the same export pass (~1300 models, ~15 minutes)
report:
	uv run python $(REPORT_SCRIPT) --output $(MODELS_MD) --ops-output $(OPS_YAML) --ops-md $(OPS_MD) \
		--exclusions $(EXCLUSIONS) --timeout $(TIMEOUT) --dynamic-timeout $(DYNAMIC_TIMEOUT)

# What CI runs. Guard solving is single-threaded sympy and a hosted runner's core is several
# times slower at it than a dev box: models whose check settles in 17s here have been measured
# blowing the 60s budget on a runner, which rewrites models.md and fails check-tree-clean.
# Excluding the known-unstable models is not enough on its own, so CI also runs at 4x budget,
# which puts its decision boundary in the same place as the one the exclusion list was
# generated at.
report.ci:
	$(MAKE) report TIMEOUT=480 DYNAMIC_TIMEOUT=240

# Regenerate export-exclusions.yaml: run every model's dynamic check for real (no exclusions
# applied) and list the ones that run out of budget. Run this locally, and commit the result
# along with the reports it rewrites.
report.exclusions:
	uv run python $(REPORT_SCRIPT) --output $(MODELS_MD) --ops-output $(OPS_YAML) --ops-md $(OPS_MD) \
		--exclusions $(EXCLUSIONS) --write-exclusions \
		--timeout $(TIMEOUT) --dynamic-timeout $(DYNAMIC_TIMEOUT)

# Quick smoke-test of the report pipeline against a handful of models. Writes to throwaway
# paths so it never leaves the committed reports half-regenerated.
report.dry-run:
	uv run python $(REPORT_SCRIPT) --limit 20 --workers 4 --output $(ROOT)/.report-dry-run.md \
		--ops-output $(ROOT)/.report-dry-run.yaml --ops-md $(ROOT)/.report-dry-run.ops.md

# ── CI helpers ────────────────────────────────────────────────────────────────

# Fail with a diff if the working tree has uncommitted changes (used in CI to
# catch generated files, like models.md, drifting from what's committed)
check-tree-clean:
	@status="$$(git status --porcelain)"; \
	if [ -n "$$status" ]; then \
		echo "$$status"; \
		git diff; \
		exit 1; \
	fi
