"""Regression coverage for the release-tier eligibility rule in `selection.render_manifest`.

vit_small_patch16_dinov3_qkvb.lvd1689m slipped into models-selected.yaml as `release: true`
because eligibility only checked for a fetchable hub_id and a weight cap -- it is a
self-supervised DINOv3 backbone with `num_classes=0` by design (no classification head), so
`export_pt2`'s worker built a contract.json with `classes.count: 0`, which fails
contract.schema.json's `minimum: 1` and hard-stopped `make release` at tag-push time, not on
any commit before it. These tests pin the fix -- a hub-fetchable, in-cap-weight model with no
classification head must never be marked release-eligible -- independent of which model
happens to occupy a candidate slot on any given run.
"""
from pt2_export_core.selection import render_manifest


def _candidate(weight_mb=1.0):
    return {'family': 'fam', 'nodes': 10, 'ops': set(), 'weight_mb': weight_mb,
            'resolution': '224x224'}


def _render(num_classes, weight_mb=1.0, hub_id='timm/m'):
    candidates = {'m': _candidate(weight_mb=weight_mb)}
    text, summary = render_manifest(
        selected=[('m', 'include')], candidates=candidates, target=1, max_nodes=1000,
        max_weight_mb=150.0, release_max_weight_mb=100.0, include=['m'], exclude=[],
        pretrained_info=lambda name: ('tag', hub_id, num_classes) if hub_id else (None, None, None),
    )
    return text, summary


def test_headless_backbone_with_fetchable_weights_is_not_release_eligible():
    text, summary = _render(num_classes=0)
    assert 'release: false' in text
    assert summary['release_models'] == 0


def test_classifier_with_fetchable_weights_is_release_eligible():
    text, summary = _render(num_classes=1000)
    assert 'release: true' in text
    assert summary['release_models'] == 1


def test_headless_backbone_ineligible_even_when_hub_id_and_weight_cap_pass():
    # Isolates num_classes as the deciding factor: hub_id present and weight well under cap,
    # so the only thing that can make this ineligible is num_classes=0.
    text, _summary = _render(num_classes=0, weight_mb=1.0, hub_id='timm/m')
    assert 'release: false' in text


def test_no_hub_weights_is_not_release_eligible_regardless_of_num_classes():
    text, summary = _render(num_classes=1000, hub_id=None)
    assert 'release: false' in text
    assert summary['release_models'] == 0
