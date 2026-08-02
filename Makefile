ROOT          := $(CURDIR)
SCRIPTS_DIR   := $(ROOT)/scripts
REPORT_SCRIPT     := $(SCRIPTS_DIR)/export_report.py
SELECT_SCRIPT     := $(SCRIPTS_DIR)/select_models.py
PT2_SCRIPT        := $(SCRIPTS_DIR)/export_pt2.py
POPULARITY_SCRIPT := $(SCRIPTS_DIR)/fetch_popularity.py
CURVE_SCRIPT      := $(SCRIPTS_DIR)/coverage_curve.py
MODELS_MD     := $(ROOT)/models.md
OPS_YAML      := $(ROOT)/ops.yaml
OPS_MD        := $(ROOT)/ops.md
EXCLUSIONS    := $(ROOT)/export-exclusions.yaml
MANIFEST      := $(ROOT)/models-selected.yaml
POPULARITY    := $(ROOT)/model-popularity.yaml
CURVE         := $(ROOT)/coverage-curve.yaml
DIFFERENCES   := $(ROOT)/graph-differences.yaml
MODELS_DIR    := $(ROOT)/models
BUILD_DIR     := $(ROOT)/.build
DATA_DIR      := $(ROOT)/data

# Per-model subprocess budget and the dynamic-shape sub-check budget. Both are wall-clock,
# so both have to move together: a dynamic budget above the subprocess one would just get
# the whole worker killed, losing that model's params, FLOPs and operators too.
TIMEOUT          ?= 120
DYNAMIC_TIMEOUT  ?= 60

.PHONY: report report.ci report.exclusions report.dry-run check-tree-clean \
        models models.select models.popularity models.curve models.dry-run models.verify models.differences \
        models.fetch download images release check-models

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

# ── PT2 graphs ───────────────────────────────────────────────────────────────

# Refresh model-popularity.yaml from the HuggingFace Hub (one paginated API call, a few
# seconds). Needs network; not part of `models.select` so that step stays offline. Rerun
# occasionally -- popularity drifts, the committed reports it feeds do not need to.
models.popularity:
	uv run python $(POPULARITY_SCRIPT) --output $(POPULARITY)

# Recompute which models to publish, from the committed reports. Runs in seconds -- it reads
# ops.yaml/models.md/model-popularity.yaml rather than exporting anything -- so the subset is
# reviewable in a diff before any archive is built. `include`/`exclude` in the manifest are
# preserved. Pass TARGET= to override the model count, e.g. `make models.select TARGET=120` --
# see coverage-curve.yaml (make models.curve) for what count buys what coverage. 100 is the
# knee of that curve: op-config coverage gain per 10 models drops from ~5-7pp to ~2.5pp around
# here, while committed size keeps climbing linearly (~4KB/node) regardless of where it bends.
TARGET ?= 100
models.select:
	uv run python $(SELECT_SCRIPT) --models-md $(MODELS_MD) --ops $(OPS_YAML) \
		--popularity $(POPULARITY) --target $(TARGET) --output $(MANIFEST)

# Report (operator, configuration) coverage and family breadth at every model count in steps
# of 10, from 10 up to where the selection saturates on its own. Runs in seconds, same inputs
# as models.select -- read coverage-curve.yaml to decide TARGET before committing to it.
models.curve:
	uv run python $(CURVE_SCRIPT) --models-md $(MODELS_MD) --ops $(OPS_YAML) \
		--popularity $(POPULARITY) --output $(CURVE)

# Export every selected model and commit the JSON parts of its .pt2 (~100 models, a few minutes).
# Random weights, fully offline: the graph does not depend on what the tensors contain, and
# keeping this path offline is what makes it reproducible in CI.
models:
	uv run python $(PT2_SCRIPT) --manifest $(MANIFEST) --models-dir $(MODELS_DIR) \
		--build-dir $(BUILD_DIR) build

# Smoke-test the graph pipeline on a handful of models, into a throwaway directory, so it
# never leaves models/ half-regenerated.
models.dry-run:
	uv run python $(PT2_SCRIPT) --manifest $(MANIFEST) --models-dir $(ROOT)/.models-dry-run \
		--build-dir $(BUILD_DIR) build --limit 3

# Hold every committed graph to the operator counts ops.yaml recorded for the same model.
# The 50 that agree exactly are evidence the published graph is the one the reports describe;
# the rest are pinned in graph-differences.yaml, and a change either way fails.
models.verify:
	uv run python $(PT2_SCRIPT) --manifest $(MANIFEST) --models-dir $(MODELS_DIR) \
		verify --ops $(OPS_YAML) --differences $(DIFFERENCES)

# Re-record graph-differences.yaml. Run after a torch or timm bump moves a decomposition,
# and review the diff -- a model appearing or vanishing there is worth understanding.
models.differences:
	uv run python $(PT2_SCRIPT) --manifest $(MANIFEST) --models-dir $(MODELS_DIR) \
		verify --ops $(OPS_YAML) --differences $(DIFFERENCES) --write

# ── Release ──────────────────────────────────────────────────────────────────

# Sample images and ImageNet labels, unmodified from their upstream releases.
download:
	bash $(SCRIPTS_DIR)/download.sh $(DATA_DIR)

# Warm a shared HuggingFace cache with the release tier's checkpoints. The only target that
# reaches the network for weights; everything else runs with HF_HUB_OFFLINE=1 against
# whatever this left behind.
models.fetch:
	uv run python $(PT2_SCRIPT) --manifest $(MANIFEST) fetch

# One shared images archive, plus one archive per release-tier model holding its .pt2,
# its preprocessing recipe and the predictions it should reproduce on those images.
images: $(BUILD_DIR)/images.zip

$(BUILD_DIR)/images.zip: FORCE | download
	@mkdir -p $(BUILD_DIR)
	rm -f $@
	cd $(DATA_DIR) && zip -q -X -r $@ images labels SOURCES.md -x 'images/.*'

release: $(BUILD_DIR)/images.zip
	uv run python $(PT2_SCRIPT) --manifest $(MANIFEST) --build-dir $(BUILD_DIR) \
		release --images $(DATA_DIR)/images

# Single-model helpers, e.g. `make resnet10t.convert`. FORCE makes the pattern rules always
# run, which is what .PHONY would do if it applied to patterns.
FORCE:

%.convert: FORCE
	uv run python $(PT2_SCRIPT) convert $* --output $(BUILD_DIR)/$*.pt2

%.extract: FORCE
	uv run python $(PT2_SCRIPT) --models-dir $(MODELS_DIR) extract $* --pt2 $(BUILD_DIR)/$*.pt2

# Fetches just this model's weights first: the export worker itself is offline, so the
# checkpoint has to already be in the cache by the time it runs.
%.release: FORCE $(BUILD_DIR)/images.zip
	uv run python $(PT2_SCRIPT) --manifest $(MANIFEST) fetch --only $*
	uv run python $(PT2_SCRIPT) --manifest $(MANIFEST) --build-dir $(BUILD_DIR) \
		release --images $(DATA_DIR)/images --workers 1 --only $*

# ── CI helpers ────────────────────────────────────────────────────────────────

# Readable diff for models/: the committed JSON is minified onto one line, so `git diff`
# alone shows a single changed line and says nothing about what actually moved.
check-models:
	bash $(SCRIPTS_DIR)/check_models.sh $(MODELS_DIR)
# Fail with a diff if the working tree has uncommitted changes (used in CI to
# catch generated files, like models.md, drifting from what's committed)
check-tree-clean:
	@status="$$(git status --porcelain)"; \
	if [ -n "$$status" ]; then \
		echo "$$status"; \
		git diff; \
		exit 1; \
	fi
