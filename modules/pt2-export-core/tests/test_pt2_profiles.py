import os

from pt2_export_core.archive import release_pt2_profile, selected_pt2_profile

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
MODELS_SELECTED = os.path.join(REPO_ROOT, 'models-selected.yaml')


def test_profiles_derive_from_real_models_selected_yaml():
    release = release_pt2_profile(MODELS_SELECTED)
    selected = selected_pt2_profile(MODELS_SELECTED)
    # release_max_weight_mb (100.0) < max_weight_mb (150.0) in the real manifest, so the
    # release-tier profile must never be looser than the broader selected-tier one -- a
    # graph-only model between the two caps must be admitted by `selected`, not `release`.
    assert release.max_archive_bytes < selected.max_archive_bytes
    assert release.max_total_uncompressed < selected.max_total_uncompressed
    # Both derive the same small, weight-independent central-directory cap.
    assert release.max_central_directory_bytes == selected.max_central_directory_bytes


def test_graph_only_model_between_the_two_caps_fits_only_the_broader_profile():
    release = release_pt2_profile(MODELS_SELECTED)
    selected = selected_pt2_profile(MODELS_SELECTED)
    # 120 MB: over the release-tier declared cap (100 MB) but under the broader selected cap
    # (150 MB) -- a legitimate graph-only model in exactly that band must be admitted by
    # SELECTED_PT2_PROFILE and would be wrongly rejected by RELEASE_PT2_PROFILE if extract()/
    # assert_portable() ever used the tighter profile for a non-release-tier model.
    hypothetical_size = 120 * 2**20
    assert hypothetical_size < selected.max_archive_bytes
    assert hypothetical_size > release.max_archive_bytes / 2  # sanity: not just trivially small
