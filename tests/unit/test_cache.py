"""Unit tests for baton.model.cache (PRD 5.4)."""

from __future__ import annotations

import json

import pytest
import torch

from baton.model import safetensors_io as sio
from baton.model.cache import (
    CACHE_ENV,
    CacheError,
    Manifest,
    ModelCache,
    default_root,
    source_digest,
)
from baton.model.quant import quantize

REPO = "meta-llama/Llama-3.2-1B"
REV = "main"


@pytest.fixture
def cache(tmp_path):
    return ModelCache(REPO, REV, tmp_path)


def layer_weights(indices, tier="int4", out=64, in_f=256):
    torch.manual_seed(0)
    return {
        f"model.layers.{i}.{proj}.weight": quantize(torch.randn(out, in_f) * 0.02, tier)
        for i in indices
        for proj in ("self_attn.q_proj", "mlp.down_proj")
    }


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #


def test_paths_follow_the_prd_layout(cache, tmp_path):
    assert cache.dir == tmp_path / "models" / REPO / REV
    assert cache.range_path(0, 8, "int4").name == "0-8.int4.safetensors"
    assert cache.manifest_path(0, 8, "int4").name == "0-8.int4.json"
    assert cache.head_path("int8").name == "head.int8.safetensors"
    assert cache.embed_path.name == "embed.bf16.bin"


def test_the_cache_root_honours_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(CACHE_ENV, str(tmp_path / "elsewhere"))
    assert default_root() == tmp_path / "elsewhere"


def test_the_default_root_is_under_the_home_cache(monkeypatch):
    monkeypatch.delenv(CACHE_ENV, raising=False)
    assert default_root().parts[-2:] == (".cache", "baton")


# --------------------------------------------------------------------------- #
# Write and read back
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("tier", ["bf16", "int8", "int4"])
def test_a_range_round_trips(cache, tier):
    weights = layer_weights([0, 1], tier)
    cache.put_range(0, 2, tier, weights)
    back = cache.read_range(0, 2, tier)

    assert set(back) == set(weights)
    for name, qw in weights.items():
        assert back[name].tier == qw.tier
        assert back[name].shape == qw.shape
        assert torch.equal(back[name].dequantize(), qw.dequantize())


def test_norms_and_biases_ride_along_unquantized(cache):
    extra = {
        "model.layers.0.input_layernorm.weight": torch.ones(64, dtype=torch.bfloat16),
        "model.layers.0.self_attn.q_proj.bias": torch.zeros(64, dtype=torch.bfloat16),
    }
    cache.put_range(0, 1, "int4", layer_weights([0]), extra)
    raw = sio.read_header(cache.range_path(0, 1, "int4"))[0]
    for name in extra:
        assert name in raw


def test_an_extra_tensor_may_not_shadow_a_weight(cache):
    weights = layer_weights([0])
    name = next(iter(weights))
    with pytest.raises(CacheError):
        cache.put_range(0, 1, "int4", weights, {f"{name}.qweight": torch.ones(2)})


def test_reading_an_absent_range_raises(cache):
    with pytest.raises(CacheError):
        cache.read_range(0, 2, "int4")


# --------------------------------------------------------------------------- #
# The hit rule (PRD 5.4)
# --------------------------------------------------------------------------- #


def test_an_exact_range_is_a_hit(cache):
    cache.put_range(0, 8, "int4", layer_weights([0]))
    assert cache.has_range(0, 8, "int4")


def test_a_different_range_is_a_miss(cache):
    cache.put_range(0, 8, "int4", layer_weights([0]))
    assert not cache.has_range(0, 7, "int4")
    assert not cache.has_range(1, 8, "int4")


def test_a_different_quant_is_a_miss(cache):
    cache.put_range(0, 8, "int4", layer_weights([0]))
    assert not cache.has_range(0, 8, "int8")


def test_a_different_revision_is_a_miss(cache, tmp_path):
    cache.put_range(0, 8, "int4", layer_weights([0]))
    assert not ModelCache(REPO, "other-rev", tmp_path).has_range(0, 8, "int4")


def test_a_changed_source_digest_is_a_miss(cache):
    cache.put_range(0, 8, "int4", layer_weights([0]), digest="aaa")
    assert cache.has_range(0, 8, "int4", digest="aaa")
    assert not cache.has_range(0, 8, "int4", digest="bbb")


def test_a_missing_data_file_is_a_miss(cache):
    cache.put_range(0, 8, "int4", layer_weights([0]))
    cache.range_path(0, 8, "int4").unlink()
    assert not cache.has_range(0, 8, "int4")


def test_a_corrupt_manifest_is_a_miss_not_a_crash(cache):
    cache.put_range(0, 8, "int4", layer_weights([0]))
    cache.manifest_path(0, 8, "int4").write_text("{ truncated")
    assert not cache.has_range(0, 8, "int4")
    assert cache.manifests() == []


def test_source_digest_changes_with_the_byte_range():
    a = sio.TensorRef("w", "f", "BF16", (2, 2), 0, 8)
    b = sio.TensorRef("w", "f", "BF16", (2, 2), 8, 16)
    assert source_digest([a]) == source_digest([a])
    assert source_digest([a]) != source_digest([b])


def test_source_digest_ignores_the_order(cache):
    a = sio.TensorRef("a", "f", "BF16", (2, 2), 0, 8)
    b = sio.TensorRef("b", "f", "BF16", (2, 2), 8, 16)
    assert source_digest([a, b]) == source_digest([b, a])


# --------------------------------------------------------------------------- #
# Per-name reuse across ranges — the re-partition case
# --------------------------------------------------------------------------- #


def test_a_wider_cached_range_seeds_a_narrower_one_with_no_download(cache):
    """PRD 5.4: a cached 10-30 seeds a new 10-25 by copying."""
    wide = layer_weights(range(10, 30))
    cache.put_range(10, 30, "int4", wide)

    wanted = list(layer_weights(range(10, 25)))
    found, missing = cache.gather(wanted, "int4")

    assert missing == []
    assert set(found) == set(wanted)
    for name in wanted:
        assert torch.equal(found[name].dequantize(), wide[name].dequantize())


def test_gather_reports_the_layers_it_lacks(cache):
    cache.put_range(10, 20, "int4", layer_weights(range(10, 20)))
    wanted = list(layer_weights(range(15, 25)))
    found, missing = cache.gather(wanted, "int4")

    assert len(found) == 10  # layers 15..19, two weights each
    assert len(missing) == 10  # layers 20..24
    assert all("layers.2" in n for n in missing)


def test_gather_stitches_two_cached_ranges(cache):
    cache.put_range(0, 4, "int4", layer_weights(range(4)))
    cache.put_range(4, 8, "int4", layer_weights(range(4, 8)))
    wanted = list(layer_weights(range(2, 6)))
    found, missing = cache.gather(wanted, "int4")
    assert missing == []
    assert set(found) == set(wanted)


def test_gather_ignores_a_range_in_another_tier(cache):
    cache.put_range(0, 4, "int8", layer_weights(range(4), "int8"))
    _, missing = cache.gather(list(layer_weights(range(4))), "int4")
    assert len(missing) == 8


def test_gather_on_an_empty_cache_reports_everything_missing(cache):
    wanted = list(layer_weights([0]))
    found, missing = cache.gather(wanted, "int4")
    assert found == {}
    assert missing == wanted


def test_locate_names_the_range_that_holds_each_weight(cache):
    cache.put_range(0, 4, "int4", layer_weights(range(4)))
    where = cache.locate(["model.layers.2.mlp.down_proj.weight"], "int4")
    assert where["model.layers.2.mlp.down_proj.weight"] == (0, 4)


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #


def test_the_manifest_records_the_quant_parameters(cache):
    cache.put_range(0, 2, "int4", layer_weights([0, 1]), digest="deadbeef")
    man = Manifest.from_json(cache.manifest_path(0, 2, "int4").read_text())
    assert man.repo_id == REPO and man.revision == REV
    assert man.quant == "int4" and man.source_sha256 == "deadbeef"
    meta = man.weights["model.layers.0.mlp.down_proj.weight"]
    assert meta.tier == "int4" and meta.group == 128 and meta.shape == (64, 256)


def test_the_manifest_is_valid_json_on_disk(cache):
    cache.put_range(0, 1, "int4", layer_weights([0]))
    json.loads(cache.manifest_path(0, 1, "int4").read_text())


def test_a_manifest_round_trips_through_json():
    man = Manifest(REPO, REV, 0, 4, "int8", source_sha256="x")
    assert Manifest.from_json(man.to_json()) == man


# --------------------------------------------------------------------------- #
# Head metadata
# --------------------------------------------------------------------------- #


def test_metadata_round_trips(cache):
    cache.put_meta("config.json", {"n_layers": 16})
    cache.put_meta("headers/model-00001.json", {"w": {"dtype": "BF16"}})
    assert cache.get_meta("config.json")["n_layers"] == 16
    assert cache.has_meta("headers/model-00001.json")


def test_reading_absent_metadata_raises(cache):
    with pytest.raises(CacheError):
        cache.get_meta("index.json")


# --------------------------------------------------------------------------- #
# Embedding table (PRD 5.3, decision D5)
# --------------------------------------------------------------------------- #


def test_the_embedding_table_memmaps_and_rows_match(cache):
    torch.manual_seed(0)
    table = (torch.randn(100, 32) * 0.02).to(torch.bfloat16)
    cache.put_embed(table)

    mm = cache.open_embed(100, 32)
    rows = ModelCache.embed_rows(mm, [0, 7, 99])
    assert rows.shape == (3, 32)
    assert torch.equal(rows, table[[0, 7, 99]])


def test_the_embedding_file_is_exactly_two_bytes_per_element(cache):
    cache.put_embed(torch.zeros(100, 32, dtype=torch.bfloat16))
    assert cache.embed_path.stat().st_size == 100 * 32 * 2


def test_a_wrong_sized_embedding_file_raises(cache):
    cache.put_embed(torch.zeros(100, 32, dtype=torch.bfloat16))
    with pytest.raises(CacheError):
        cache.open_embed(101, 32)


def test_opening_an_absent_embedding_raises(cache):
    with pytest.raises(CacheError):
        cache.open_embed(10, 4)


def test_a_float32_table_is_cast_to_bf16(cache):
    cache.put_embed(torch.zeros(8, 4, dtype=torch.float32))
    assert cache.embed_path.stat().st_size == 8 * 4 * 2


# --------------------------------------------------------------------------- #
# Housekeeping
# --------------------------------------------------------------------------- #


def test_disk_bytes_counts_the_revision(cache):
    assert cache.disk_bytes() == 0
    cache.put_range(0, 2, "int4", layer_weights([0, 1]))
    assert cache.disk_bytes() > 0


def test_evict_removes_both_files_and_is_idempotent(cache):
    cache.put_range(0, 2, "int4", layer_weights([0, 1]))
    cache.evict_range(0, 2, "int4")
    assert not cache.has_range(0, 2, "int4")
    cache.evict_range(0, 2, "int4")


def test_no_part_files_are_left_behind(cache):
    cache.put_range(0, 2, "int4", layer_weights([0, 1]))
    cache.put_meta("config.json", {})
    assert not list(cache.dir.rglob("*.part"))


def test_int4_uses_about_a_quarter_of_the_bf16_disk(tmp_path):
    weights_bf16 = layer_weights(range(4), "bf16", out=256, in_f=512)
    weights_int4 = layer_weights(range(4), "int4", out=256, in_f=512)
    a = ModelCache(REPO, REV, tmp_path / "a")
    b = ModelCache(REPO, REV, tmp_path / "b")
    a.put_range(0, 4, "bf16", weights_bf16)
    b.put_range(0, 4, "int4", weights_int4)
    assert b.disk_bytes() < a.disk_bytes() * 0.32
