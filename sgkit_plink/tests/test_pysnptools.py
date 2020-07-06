import numpy as np
import pytest
from sgkit_plink import pysnptools

example_dataset_1 = "plink_sim_10s_100v_10pmiss"


@pytest.fixture(params=[dict()])
def ds1(shared_datadir, request):
    path = shared_datadir / example_dataset_1
    return pysnptools.read_plink(path, bim_sep="\t", fam_sep="\t", **request.param)


def test_read_slicing(ds1):
    gt = ds1["call/genotype"]
    shape = gt.shape
    assert gt[:3].shape == (3,) + shape[1:]
    assert gt[:, :3].shape == shape[:1] + (3,) + shape[2:]
    assert gt[:3, :5].shape == (3, 5) + shape[2:]
    assert gt[:3, :5, :1].shape == (3, 5, 1)


@pytest.mark.parametrize("ds1", [dict(bim_int_contig=True)], indirect=True)
def test_read_int_contig(ds1):
    # Test contig parse as int (the value is always "1" in .bed for ds1)
    assert np.all(ds1["variant/contig"].values == 1)
    assert ds1.attrs["contigs"] == ["1"]


@pytest.mark.parametrize("ds1", [dict(bim_int_contig=False)], indirect=True)
def test_read_str_contig(ds1):
    # Test contig indexing as string (the value is always "1" in .bed for ds1)
    assert np.all(ds1["variant/contig"].values == 0)
    assert ds1.attrs["contigs"] == ["1"]


def test_read_stat_call_rate(ds1):
    # Validate call rate for each sample
    sample_call_rates = (
        (ds1["call/genotype"] >= 0).max(dim="ploidy").mean(dim="variants").values
    )
    np.testing.assert_equal(
        sample_call_rates, [0.95, 0.9, 0.91, 0.87, 0.86, 0.83, 0.86, 0.87, 0.92, 0.92]
    )


def test_read_stat_alt_alleles(ds1):
    # Validate alt allele sum for each sample
    n_alt_alleles = (
        ds1["call/genotype"].clip(0, 2).sum(dim="ploidy").sum(dim="variants").values
    )
    np.testing.assert_equal(n_alt_alleles, [102, 95, 98, 94, 88, 91, 90, 98, 96, 103])
