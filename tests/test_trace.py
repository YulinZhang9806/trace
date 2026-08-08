"""Testing suite for TRACE HMM functions."""

import msprime as msp
import numpy as np 
import tskit
import pytest

from tracehmm import TRACE

ts1 = msp.sim_ancestry(
    samples=100,
    population_size=1e4,
    recombination_rate=1e-8,
    random_seed=42,
    sequence_length=5e6,
)
ts2 = msp.sim_ancestry(
    samples=50,
    population_size=1e4,
    recombination_rate=5e-8,
    random_seed=24,
    sequence_length=5e6,
)


def test_init():
    """Test that trace can be naively initialized."""
    hmm = TRACE()
    assert hmm is not None


@pytest.mark.parametrize("ts", [ts1, ts2])
def test_add_ts(ts):
    """Test that adding a tree-sequence works fine."""
    hmm = TRACE()
    hmm.add_tree_sequence(ts)
    assert hmm.ts is not None
    assert hmm.ts.num_samples > 0


@pytest.mark.parametrize("ts", [ts1, ts2])
def test_extract_ncoal(ts):
    """Test extraction of number of coalescent events from TRACE."""
    hmm = TRACE()
    hmm.add_tree_sequence(ts)
    # NOTE: there is some funkiness about the random seed setting here ...
    ncoal, t1s, t2s, n_leaves = hmm.extract_ncoal(idx=0, t_archaic=15e3)
    assert ncoal.size > 0


@pytest.mark.parametrize("ts", [ts1, ts2])
def test_extract_ncoal_bad_idx(ts):
    """Test extraction of number of coalescent events from TRACE."""
    hmm = TRACE()
    hmm.add_tree_sequence(ts)
    # NOTE: there is some funkiness about the random seed setting here ...
    assert hmm.ts is not None
    x = int(2 * hmm.ts.num_samples + 1)
    with pytest.raises(ValueError):
        ncoal, t1s, t2s, n_leaves = hmm.extract_ncoal(idx=x, t_archaic=15e3)


@pytest.mark.parametrize("ts", [ts1, ts2])
@pytest.mark.parametrize("t", [-100, 0.0, "x"])
def test_extract_ncoal_bad_time(ts, t):
    """Test extraction of number of coalescent events from TRACE."""
    hmm = TRACE()
    hmm.add_tree_sequence(ts)
    assert hmm.ts is not None
    with pytest.raises((AssertionError, TypeError)):
        ncoal, t1s, t2s, n_leaves = hmm.extract_ncoal(idx=0, t_archaic=t)


def test_masking_function(tmp_path_factory):
    """Test out the behavior of TRACE masking function."""
    hmm = TRACE()
    # Make a simple test case here 
    chrom = 'chr1'
    seq_length = 400
    tables = tskit.TableCollection(sequence_length=seq_length)
    sample = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    spans = [(0, 250), (250, 300), (300, 400)]
    for left, right in spans:
        root = tables.nodes.add_row(time=1)
        tables.edges.add_row(left=left, right=right, parent=root, child=sample)
    tables.sort()
    ts = tables.tree_sequence()
    bp = ts.breakpoints(as_array=True).astype(int)
    treespan = np.column_stack([bp[:-1], bp[1:]])
    fn1 = tmp_path_factory.mktemp("extract_data") / "chr1_regions.bed"
    with open(fn1, "w+") as out:
        out.write("chr1\t0\t255\n")
        out.write("chr1\t300\t360\n")
    fn2 = tmp_path_factory.mktemp("extract_data") / "chr2_regions.bed"
    with open(fn2, "w+") as out:
        out.write("chr2\t0\t10000\n")
    output_mask  = hmm.mask_regions(treespan=treespan, chrom=chrom, maskfile=fn1, f=0.99)
    assert output_mask.size  == treespan.shape[0]
    assert output_mask[0] == 1
    assert output_mask[1] == 0
    assert output_mask[2] == 0
    output_mask  = hmm.mask_regions(treespan=treespan, chrom=chrom, maskfile=fn1, f=0.5)
    assert output_mask.size  == treespan.shape[0]
    assert output_mask[0] == 1
    assert output_mask[1] == 0
    assert output_mask[2] == 1
    # with f=None, any intersection counts ... 
    output_mask  = hmm.mask_regions(treespan=treespan, chrom=chrom, maskfile=fn1)
    assert output_mask.size  == treespan.shape[0]
    assert output_mask[0] == 1
    assert output_mask[1] == 1
    assert output_mask[2] == 1
    output_mask  = hmm.mask_regions(treespan=treespan, chrom=chrom, maskfile=fn2, f=0.25)
    assert output_mask.size  == treespan.shape[0]
    assert output_mask[0] == 0
    assert output_mask[1] == 0
    assert output_mask[2] == 0