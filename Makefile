ROOT          := $(CURDIR)
SCRIPTS_DIR   := $(ROOT)/scripts
REPORT_SCRIPT     := $(SCRIPTS_DIR)/export_report.py
SELECT_SCRIPT     := $(SCRIPTS_DIR)/select_models.py
PT2_SCRIPT        := $(SCRIPTS_DIR)/export_pt2.py
POPULARITY_SCRIPT := $(SCRIPTS_DIR)/fetch_popularity.py
CURVE_SCRIPT      := $(SCRIPTS_DIR)/coverage_curve.py
MODELS_MD     := $(ROOT)/models.md
# The published dialect and the functionalized one each get a single cross-reference, both being
# fixed before dispatch; core ATen gets one per backend, because that decomposition runs after
# dispatch and so depends on where the model was traced.
OPS_ATEN_YAML := $(ROOT)/ops-aten.yaml
OPS_ATEN_MD   := $(ROOT)/ops-aten.md
OPS_FUNC_YAML := $(ROOT)/ops-func.yaml
OPS_FUNC_MD   := $(ROOT)/ops-func.md
OPS_CORE      := $(ROOT)/ops-core
CORE_BACKENDS := meta cpu
OPS_CORE_YAML := $(foreach b,$(CORE_BACKENDS),$(OPS_CORE)-$(b).yaml)
EXCLUSIONS    := $(ROOT)/export-exclusions.yaml
MANIFEST      := $(ROOT)/models-selected.yaml
POPULARITY    := $(ROOT)/model-popularity.yaml
CURVE         := $(ROOT)/coverage-curve.yaml
DIFFERENCES   := $(ROOT)/graph-differences.yaml
MODELS_DIR    := $(ROOT)/models
BUILD_DIR     := $(ROOT)/.build
DATA_DIR      := $(ROOT)/data
HISTORY       := $(ROOT)/models-history.yaml

# Published output (every model .zip, images.zip, manifest.json/catalogue.json/
# compat-report.json/checksums.txt) vs. scratch/diagnostics (the .pt2 each pack step reads
# and discards, pack-results.json). release_assets() only ever scans RELEASE_DIR, which is
# what makes "no extra file in the release set" true by construction.
RELEASE_DIR   := $(BUILD_DIR)/release
WORK_DIR      := $(BUILD_DIR)/work

# Explicit, never inferred silently: manifest.json/catalogue.json need one (repo, tag, commit)
# triple every URL they build is threaded through. The shell defaults below are a convenience
# for a local run against the current checkout; release.yml overrides all three from the
# workflow's own context instead of trusting a local guess.
RELEASE_REPO   ?= $(shell git config --get remote.origin.url 2>/dev/null | sed -E 's#\.git$$##' | sed -E 's#^(https://github\.com/|git@github\.com:)##')
RELEASE_TAG    ?= $(shell git describe --tags --exact-match 2>/dev/null || echo dev)
RELEASE_COMMIT ?= $(shell git rev-parse HEAD)
DEFAULT_MODEL  ?= convit_tiny

# Per-model subprocess budget and the dynamic-shape sub-check budget. Both are wall-clock,
# so both have to move together: a dynamic budget above the subprocess one would just get
# the whole worker killed, losing that model's params, FLOPs and operators too.
TIMEOUT          ?= 120
DYNAMIC_TIMEOUT  ?= 60

.PHONY: report report.ci report.exclusions report.dry-run check-tree-clean \
        models models.select models.popularity models.curve models.dry-run models.verify models.differences \
        models.fetch models.compat-static models.role-candidates download images release release.manifest \
        release.assets release.dry-run check-history check-models test

# ── timm export report ───────────────────────────────────────────────────────

# Regenerate models.md (torch.export compatibility, weight size, FLOPs) plus the aten op
# cross-references ops-aten.*, ops-func.* and ops-core-<backend>.*, all from the same export pass
# (~1300 models, ~20 minutes)
report:
	uv run python $(REPORT_SCRIPT) --output $(MODELS_MD) \
		--ops-aten-output $(OPS_ATEN_YAML) --ops-aten-md $(OPS_ATEN_MD) \
		--ops-func-output $(OPS_FUNC_YAML) --ops-func-md $(OPS_FUNC_MD) --ops-core-prefix $(OPS_CORE) \
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
	uv run python $(REPORT_SCRIPT) --output $(MODELS_MD) \
		--ops-aten-output $(OPS_ATEN_YAML) --ops-aten-md $(OPS_ATEN_MD) \
		--ops-func-output $(OPS_FUNC_YAML) --ops-func-md $(OPS_FUNC_MD) --ops-core-prefix $(OPS_CORE) \
		--exclusions $(EXCLUSIONS) --write-exclusions \
		--timeout $(TIMEOUT) --dynamic-timeout $(DYNAMIC_TIMEOUT)

# Quick smoke-test of the report pipeline against a handful of models. Writes to throwaway
# paths so it never leaves the committed reports half-regenerated.
report.dry-run:
	uv run python $(REPORT_SCRIPT) --limit 20 --workers 4 --output $(ROOT)/.report-dry-run.md \
		--ops-aten-output $(ROOT)/.report-dry-run.aten.yaml --ops-aten-md $(ROOT)/.report-dry-run.aten.md \
		--ops-func-output $(ROOT)/.report-dry-run.func.yaml --ops-func-md $(ROOT)/.report-dry-run.func.md \
		--ops-core-prefix $(ROOT)/.report-dry-run.core

# ── PT2 graphs ───────────────────────────────────────────────────────────────

# Refresh model-popularity.yaml from the HuggingFace Hub (one paginated API call, a few
# seconds). Needs network; not part of `models.select` so that step stays offline. Rerun
# occasionally -- popularity drifts, the committed reports it feeds do not need to.
models.popularity:
	uv run python $(POPULARITY_SCRIPT) --output $(POPULARITY)

# Recompute which models to publish, from the committed reports. Runs in seconds -- it reads the
# cross-references, models.md and model-popularity.yaml rather than exporting anything -- so the
# subset is reviewable in a diff before any archive is built. `include`/`exclude` in the manifest are
# preserved. Pass TARGET= to override the model count, e.g. `make models.select TARGET=120` --
# see coverage-curve.yaml (make models.curve) for what count buys what coverage. 100 is the
# knee of that curve: op-config coverage gain per 10 models drops from ~4-6pp to ~2.3pp around
# here, while committed size keeps climbing linearly (~1.8KB/node) regardless of where it bends.
TARGET ?= 100
models.select:
	uv run python $(SELECT_SCRIPT) --models-md $(MODELS_MD) \
		--ops-aten $(OPS_ATEN_YAML) --ops-func $(OPS_FUNC_YAML) --ops-core $(OPS_CORE_YAML) \
		--popularity $(POPULARITY) --target $(TARGET) --output $(MANIFEST)

# Which models were the sole/first contributor of a coverage unit worth flagging as a role
# (suggestion #6) -- always fully regenerated into models-role-candidates.yaml, never merged
# automatically into the separate, hand-maintained models-roles.yaml. Review the diff and
# copy in whatever is worth a label.
ROLE_CANDIDATES := $(ROOT)/models-role-candidates.yaml
models.role-candidates:
	uv run python $(SELECT_SCRIPT) --models-md $(MODELS_MD) \
		--ops-aten $(OPS_ATEN_YAML) --ops-func $(OPS_FUNC_YAML) --ops-core $(OPS_CORE_YAML) \
		--popularity $(POPULARITY) --target $(TARGET) --output $(MANIFEST) \
		--write-role-candidates --role-candidates-output $(ROLE_CANDIDATES)

# Report (operator, configuration) coverage and family breadth at every model count in steps
# of 10, from 10 up to where the selection saturates on its own. Runs in seconds, same inputs
# as models.select -- read coverage-curve.yaml to decide TARGET before committing to it.
models.curve:
	uv run python $(CURVE_SCRIPT) --models-md $(MODELS_MD) \
		--ops-aten $(OPS_ATEN_YAML) --ops-func $(OPS_FUNC_YAML) --ops-core $(OPS_CORE_YAML) \
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

# Hold every committed graph to the operator counts ops-func.yaml recorded for the same model.
# Agreement is evidence the published graph is the one the reports describe -- and it can be
# expected here because both sides are functional ATen. Residual device-dependent differences
# around attention views are pinned in graph-differences.yaml,
# and a change either way fails.
models.verify:
	uv run python $(PT2_SCRIPT) --manifest $(MANIFEST) --models-dir $(MODELS_DIR) \
		verify --ops $(OPS_FUNC_YAML) --differences $(DIFFERENCES)

# Re-record graph-differences.yaml. Run after a torch or timm bump moves a decomposition,
# and review the diff -- a model appearing or vanishing there is worth understanding.
models.differences:
	uv run python $(PT2_SCRIPT) --manifest $(MANIFEST) --models-dir $(MODELS_DIR) \
		verify --ops $(OPS_FUNC_YAML) --differences $(DIFFERENCES) --write

# Static graph facts (op_facts.json only) for every selected model -- seconds, no network,
# no release context. Catches a stale/missing sidecar independently of a full release build.
models.compat-static:
	uv run python $(PT2_SCRIPT) --manifest $(MANIFEST) --models-dir $(MODELS_DIR) compat-static

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
# its preprocessing recipe and the predictions it should reproduce on those images. Lives
# under RELEASE_DIR: images.zip is itself a published, checksum-listed asset.
images: $(RELEASE_DIR)/images.zip

$(RELEASE_DIR)/images.zip: FORCE | download
	@mkdir -p $(RELEASE_DIR)
	rm -f $@
	cd $(DATA_DIR) && zip -q -X -r $@ images labels SOURCES.md -x 'images/.*'

# Convert + pack every release-tier model. Fails (non-zero) if any release-tier model's
# interpreter run fails -- pack-results.json (under WORK_DIR) is still written either way,
# but nothing after this in the DAG below runs on a hard-stop.
release: $(RELEASE_DIR)/images.zip
	uv run python $(PT2_SCRIPT) --manifest $(MANIFEST) release \
		--images $(DATA_DIR)/images --release-dir $(RELEASE_DIR) --work-dir $(WORK_DIR)

# The one canonical validation/checksum DAG: release (hard-stops here on any release-tier
# failure) -> manifest + catalogue + compat-report (schema-validated) -> verify-release
# (byte-compares every embedded graph against the committed one, re-validates every
# document, and writes checksums.txt last, only once everything above has passed).
release.manifest: release
	uv run python $(PT2_SCRIPT) --manifest $(MANIFEST) manifest \
		--release-dir $(RELEASE_DIR) --work-dir $(WORK_DIR) \
		--repo $(RELEASE_REPO) --tag $(RELEASE_TAG) --commit $(RELEASE_COMMIT) \
		--default-model $(DEFAULT_MODEL)
	uv run python $(PT2_SCRIPT) --manifest $(MANIFEST) --models-dir $(MODELS_DIR) \
		verify-release --release-dir $(RELEASE_DIR)

# release_assets(): re-hashes every file checksums.txt lists against RELEASE_DIR, fails on any
# mismatch or extra/missing file, prints the verified list (checksums.txt first) plus its own
# sha256 as `checksums_sha256=<digest>`. With EXPECT_CHECKSUMS_SHA256 set, pins against that
# digest before rehashing anything, and re-checks every manifest.json archive entry against
# the freshly-computed digests -- release.yml invokes this twice (see the workflow), bound
# together by that pinned digest, never as one unpinned call reused for both purposes.
release.assets: FORCE
	@uv run python $(PT2_SCRIPT) release-assets --release-dir $(RELEASE_DIR) \
		$(if $(EXPECT_CHECKSUMS_SHA256),--expect-checksums-sha256 $(EXPECT_CHECKSUMS_SHA256),)

# Smoke-scope: ONLY=<names>, writes under .build/smoke/ -- structurally distinct from
# RELEASE_DIR, so release.assets can never pick this up by accident.
release.dry-run: $(RELEASE_DIR)/images.zip
	uv run python $(PT2_SCRIPT) --manifest $(MANIFEST) release \
		--images $(DATA_DIR)/images --images-archive $(RELEASE_DIR)/images.zip \
		--release-dir $(BUILD_DIR)/smoke/release --work-dir $(BUILD_DIR)/smoke/work \
		--only $(ONLY)

# Single-model helpers, e.g. `make resnet10t.convert`. FORCE makes the pattern rules always
# run, which is what .PHONY would do if it applied to patterns.
FORCE:

%.convert: FORCE
	uv run python $(PT2_SCRIPT) convert $* --output $(BUILD_DIR)/$*.pt2

%.extract: FORCE
	uv run python $(PT2_SCRIPT) --models-dir $(MODELS_DIR) extract $* --pt2 $(BUILD_DIR)/$*.pt2

# Fetches just this model's weights first: the export worker itself is offline, so the
# checkpoint has to already be in the cache by the time it runs.
%.release: FORCE $(RELEASE_DIR)/images.zip
	uv run python $(PT2_SCRIPT) --manifest $(MANIFEST) fetch --only $*
	uv run python $(PT2_SCRIPT) --manifest $(MANIFEST) release \
		--images $(DATA_DIR)/images --release-dir $(RELEASE_DIR) --work-dir $(WORK_DIR) \
		--workers 1 --only $*

# ── CI helpers ────────────────────────────────────────────────────────────────

# The repo's own pytest suite (scripts/tests + the pt2-export-core package's tests):
# fast/pure-logic unit tests alongside a handful of real, small-model integration tests
# (real subprocess workers, real tiny hub downloads) that exercise export_pt2.py's actual
# pack/manifest/verify pipeline end to end. Was never wired into CI -- only the expensive,
# tag-push-only `make release.manifest` run against the real release-tier set ever exercised
# that pipeline for real, so a defect reachable from it (a bad models-selected.yaml entry, a
# selection-eligibility bug) surfaced only when cutting a release, not on the commit that
# introduced it. Runs on every commit/PR now (see build.yml) so this class of regression
# fails fast instead.
test:
	uv run pytest scripts/tests modules/pt2-export-core/tests -q

# A release-tier model removed since the previous release tag must be recorded in
# models-history.yaml, or this fails naming it. First-parent only, and first-release-safe
# (passes trivially if there is no previous tag). Requires full history (fetch-depth: 0).
check-history:
	uv run python $(SCRIPTS_DIR)/models_history.py --history $(HISTORY) check

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
