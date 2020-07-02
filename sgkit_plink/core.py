import numpy as np
import dask.dataframe as dd
import dask.array as da
from pysnptools.snpreader import Bed
from pathlib import Path
from typing import Union
from sgkit import create_genotype_call_dataset

# from .api import (  # noqa: F401
#     DIM_ALLELE,
#     DIM_PLOIDY,
#     DIM_SAMPLE,
#     DIM_VARIANT,
#     create_genotype_call_dataset,
# )

PathType = Union[str, Path]


class BedReader(object):

    def __init__(self, path, shape, dtype=np.int8, count_A1=True):
        # n variants (sid = SNP id), n samples (iid = Individual id)
        n_sid, n_iid = shape
        # Initialize Bed with empty arrays for axis data, otherwise it will
        # load the bim/map/fam files entirely into memory (it does not do out-of-core for those)
        self.bed = Bed(
            str(path),
            count_A1=count_A1,
            # Array (n_sample, 2) w/ FID and IID
            iid=np.empty((n_iid, 2), dtype="str"),
            # SNP id array (n_variants)
            sid=np.empty((n_sid,), dtype="str"),
            # Contig and positions array (n_variants, 3)
            pos=np.empty((n_sid, 3), dtype="int"),
        )
        self.shape = (n_sid, n_iid, 2)
        self.dtype = dtype
        self.ndim = 3

    @staticmethod
    def _is_empty_slice(s):
        return s.start == s.stop

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            raise IndexError(f"Indexer must be tuple (received {type(idx)})")
        if len(idx) != self.ndim:
            raise IndexError(
                f"Indexer must be two-item tuple (received {len(idx)} slices)"
            )

        # This is called by dask with empty slices before trying to read any chunks, so it may need
        # to be handled separately if pysnptools is slow here
        # if all(map(BedReader._is_empty_slice, idx)):
        #     return np.empty((0, 0), dtype=self.dtype)

        arr = self.bed[idx[::-1]].read(dtype=np.float32, view_ok=False).val.T
        arr = np.ma.masked_invalid(arr)
        arr = arr.astype(self.dtype)
        # Add a ploidy dimension, so allele counts of 0, 1, 2 correspond to 00, 01, 11
        arr2 = np.empty((arr.shape[0], arr.shape[1], 2), dtype=self.dtype)
        arr2[:, :, 0] = np.where(arr == 2, 1, 0)
        arr2[:, :, 1] = np.where(arr == 0, 0, 1)
        return arr2

    def close(self):
        # This is not actually crucial since a Bed instance with no
        # in-memory bim/map/fam data is essentially just a file pointer
        # but this will still be problematic if the an array is created
        # from the same PLINK dataset many times
        self.bed._close_bed()


def read_fam(path: PathType, sep: str = "\t"):
    names = ["sample_id", "fam_id", "pat_id", "mat_id", "is_female", "phenotype"]
    return dd.read_csv(
        str(path) + ".fam", sep=sep, names=names, storage_options=dict(auto_mkdir=False)
    )


def read_bim(path: PathType, sep: str = " "):
    names = ["contig", "variant_id", "cm_pos", "pos", "a1", "a2"]
    return dd.read_csv(
        str(path) + ".bim", sep=sep, names=names, storage_options=dict(auto_mkdir=False)
    )


def read_plink(
    path: PathType,
    chunks: Union[str, int, tuple] = "auto",
    fam_sep: str = "\t",
    bim_sep: str = " ",
    count_A1: bool = True,
    lock: bool = False,
):
    # Load axis data first to determine dimension sizes
    df_fam = read_fam(path, sep=fam_sep)
    df_bim = read_bim(path, sep=bim_sep)

    # Load genotyping data
    call_gt = da.from_array(
        # Make sure to use asarray=False in order for masked arrays to propagate
        BedReader(path, (len(df_bim), len(df_fam)), count_A1=count_A1),
        chunks=chunks,
        # Lock must be true with multiprocessing dask scheduler
        # to not get pysnptools errors (it works w/ threading backend though)
        lock=lock,
        asarray=False,
        name=f"read_plink:{path}",
    )

    # TODO: either avoid computing Dask arrays, or just use Pandas
    df_bim_pd = df_bim.compute()
    df_fam_pd = df_fam.compute()

    variant_contig_names = df_bim_pd["contig"].values
    # TODO: can we get the names from somewhere in a given order? (since following sorts them)
    u, variant_contig = np.unique(variant_contig_names, return_inverse=True)

    variant_pos = df_bim_pd["pos"].values

    a1 = df_bim_pd["a1"].values
    a2 = df_bim_pd["a2"].values
    variant_alleles = np.column_stack((a1, a2))
    variant_alleles = variant_alleles.astype(np.dtype("S1"))

    variant_id = df_bim_pd["variant_id"].values
    variant_id = variant_id.astype(str)

    sample_id = df_fam_pd["sample_id"].values
    sample_id = sample_id.astype(str)

    ds = create_genotype_call_dataset(
        variant_contig,
        variant_pos,
        variant_alleles,
        sample_id,
        call_gt=call_gt,
        variant_id=variant_id,
    )
    ds = ds.assign(family_id=([DIM_SAMPLE], sample_id))
