"""Test suite for Extract CLI."""

import os
import shutil
from pathlib import Path

import msprime
import numpy as np
import pytest
import tszip
import tskit


@pytest.fixture(scope="session")
def tszip1(tmp_path_factory):
    """Create temporary tszip file."""
    ts1 = msprime.sim_ancestry(
        samples=10,
        population_size=1e4,
        recombination_rate=1e-8,
        random_seed=42,
        sequence_length=5e6,
    )
    fn = tmp_path_factory.mktemp("extract_data") / "ts1.tsz"
    tszip.compress(ts1, fn)
    return fn


@pytest.fixture(scope="session")
def dumb_file(tmp_path_factory):
    """Create a dumb file ..."""
    fn = tmp_path_factory.mktemp("extract_data") / "xxx.txt"
    with open(fn, "w+") as out:
        out.write("Hello\nWorld\n")
    return fn


@pytest.fixture(scope="session")
def proper_chr1_regions(tmp_path_factory):
    """Create a dumb file ..."""
    fn = tmp_path_factory.mktemp("extract_data") / "chr1_regions.txt"
    with open(fn, "w+") as out:
        out.write("chr1\t0\t1000000\n")
        out.write("chr1\t2000000\t4000000\n")
    return fn


@pytest.fixture(scope="session")
def bad_chr_regions(tmp_path_factory):
    """Create a dumb file ..."""
    fn = tmp_path_factory.mktemp("extract_data") / "chr1_regions_bad.txt"
    with open(fn, "w+") as out:
        out.write("chrX\t0\t1000000\tX\n")
        out.write("chrX\t2000000\t1500000\tY\n")
    return fn


def check_npz_file(fp):
    """Check an extracted npz file."""
    assert Path(fp).is_file()
    data = np.load(fp)
    keys = [
        "ncoal",
        "t1s",
        "t2s",
        "treespan",
        "marginal_treespan",
        "marginal_mask",
        "accessible_windows",
        "individuals",
    ]
    for k in data.keys():
        assert k in keys


def test_extract_ts1(tszip1, tmp_path_factory):
    """Test extraction of a standard tszip file."""
    out_fp = Path(tszip1).with_suffix(".npz")
    outfix = Path(tszip1).with_suffix("")
    exit_status = os.system(
        f"trace-extract --tree-file {tszip1} -t 15e3 --individuals 0,1,2 --out {outfix}"
    )
    assert exit_status == 0
    assert Path(out_fp).is_file()
    check_npz_file(out_fp)
    data = np.load(out_fp)
    assert data["individuals"].size == 3


def test_check_bad_ts(dumb_file):
    """Test extraction with poor file input."""
    outfix = Path(dumb_file).with_suffix("")
    exit_status = os.system(
        f"trace-extract --tree-file {dumb_file} -t 15e3 --individuals 0 --out {outfix}"
    )
    assert exit_status != 0


@pytest.mark.parametrize("indiv", ["200", "A,B,C", "-100"])
def test_bad_indivs(tszip1, indiv):
    """Test specifying bad individuals."""
    outfix = Path(tszip1).with_suffix("")
    exit_status = os.system(
        f"trace-extract --tree-file {tszip1} -t 15e3 --individuals {indiv} --out {outfix}"
    )
    assert exit_status != 0


@pytest.mark.parametrize("t", [-1, 1e20])
def test_t_archaic(tszip1, t):
    """Test different estimates of t-archaic."""
    outfix = Path(tszip1).with_suffix("")
    exit_status = os.system(
        f"trace-extract --tree-file {tszip1} --t-archaic {t} --out {outfix}"
    )
    assert exit_status != 0


@pytest.mark.parametrize("w", [100, 1000, 10000, 100000])
def test_window_size(tszip1, w):
    """Test different window sizes."""
    out_fp = Path(tszip1).with_suffix(".npz")
    outfix = Path(tszip1).with_suffix("")
    exit_status = os.system(
        f"trace-extract --tree-file {tszip1} -t 15e3 --individuals 0,1,2 --window-size {w} --out {outfix}"
    )
    assert exit_status == 0
    assert Path(out_fp).is_file()


@pytest.mark.parametrize("w", [None, 0, -100, 100.0])
def test_bad_window_size(tszip1, w):
    """Test extraction with bad window sizes."""
    outfix = Path(tszip1).with_suffix("")
    exit_status = os.system(
        f"trace-extract --tree-file {tszip1} -t 15e3 --individuals 0,1,2 --window-size {w} --out {outfix}"
    )
    assert exit_status != 0


def test_chrom_regions(tszip1, proper_chr1_regions):
    """Test defining chromosomal regions."""
    out_fp = Path(tszip1).with_suffix(".npz")
    outfix = Path(tszip1).with_suffix("")
    if shutil.which("bedtools") is None:
        exit_status = os.system(
            f"trace-extract --tree-file {tszip1} -t 15e3 --individuals 0,1,2 --chrom chr1 --include-regions {proper_chr1_regions} --out {outfix}"
        )
        assert exit_status != 0
    else:
        exit_status = os.system(
            f"trace-extract --tree-file {tszip1}  -t 15e3 --individuals 0,1,2 --chrom chr1 --include-regions {proper_chr1_regions} --out {outfix}"
        )
        assert exit_status == 0
        assert Path(out_fp).is_file()
        check_npz_file(out_fp)


@pytest.mark.parametrize("c", [None, "chr2", "chrX", "X"])
def test_chrom_mismatch(tszip1, proper_chr1_regions, c):
    """Test that chromosome mismatch is not supported."""
    outfix = Path(tszip1).with_suffix("")
    exit_status = os.system(
        f"trace-extract --tree-file {tszip1} -t 15e3 --individuals 0,1,2 --chrom {c} --include_regions {proper_chr1_regions} --out {outfix}"
    )
    assert exit_status != 0


def test_bad_chrom_regions(tszip1, bad_chr_regions):
    """Test defining chromosomal regions."""
    outfix = Path(tszip1).with_suffix("")
    exit_status = os.system(
        f"trace-extract --tree-file {tszip1}  -t 15e3 --individuals 0,1,2 --chrom chr1 --include-regions {bad_chr_regions} --out {outfix}"
    )
    assert exit_status != 0


def test_window_bug_repro():
    """
    Integrate and run the small window-aggregation pipeline from tracehmm.fix.
    Thanks @Jie Wang for reporting this bug and providing a minimal test case.
    """
    # TRACE's windowed aggregation from extract_cli.py::get_data (lines 64-143)
    def trace_windowed(treespan, feat, windowsize, seq_length):
        treespan = treespan.astype(int)
        genome_length = seq_length
        m = int(genome_length / windowsize) + int(genome_length % windowsize > 0)
        ind = np.array([0])
        ncoal_sub = np.zeros((1, m))
        accessible_windows = np.ones(m)
        tncoal = np.array([feat], dtype=float)# shape (1, n_trees)
        mask = np.ones(treespan.shape[0])
        t = 0
        curtrees = []
        for k in range(m):
            while t < treespan.shape[0] and treespan[t][0] < int(k * windowsize + windowsize):
                if mask[t] == 1:
                    curtrees.append(t)
                else:
                    curtrees.append(-1)
                t += 1
            if len(curtrees) == 0:
                for i in range(len(ind)):
                    ncoal_sub[i][k] = tncoal[i][t - 1]
            else:
                treelens = []
                curtrees = np.array(curtrees)
                curtrees = curtrees[curtrees >= 0]
                if len(curtrees) == 0:
                    accessible_windows[k] = 0
                    for i in range(len(ind)):
                        ncoal_sub[i][k] = 1e-10
                else:
                    for j in range(len(curtrees)):
                        treelens.append(
                            min(treespan[curtrees[j]][1], int(k * windowsize + windowsize))
                            - max(treespan[curtrees[j]][0], int(k * windowsize))
                        )
                    treelens = np.array(treelens)
                    curtrees = curtrees[treelens > 1]
                    treelens = treelens[treelens > 1]
                    if len(curtrees) == 0:
                        accessible_windows[k] = 0
                        for i in range(len(ind)):
                            ncoal_sub[i][k] = 1e-10
                    else:
                        for i in range(len(ind)):
                            ncoal_sub[i][k] = np.average(tncoal[i][curtrees], weights=treelens)
                curtrees = []
                if treespan[t - 1][1] > (k + 1) * windowsize:  # BUG: carry-over
                    if mask[t - 1] == 1:
                        curtrees.append(t - 1)
                    else:
                        curtrees.append(-1)
        return ncoal_sub[0], accessible_windows
    
    def reference_windowed(treespan, feat, windowsize, seq_length):
        treespan = treespan.astype(float)
        m = int(seq_length / windowsize) + int(seq_length % windowsize > 0)
        bounds = np.array([k * windowsize for k in range(m)] + [seq_length], dtype=float)
        out = np.zeros(m)
        accessible = np.zeros(m)
        for k in range(m):
            wl, wr = bounds[k], bounds[k + 1]
            num, den = 0.0, 0.0
            for ti in range(treespan.shape[0]):
                ov = min(treespan[ti][1], wr) - max(treespan[ti][0], wl)
                if ov > 0:
                    num += feat[ti] * ov
                    den += ov
            if den > 0:
                out[k] = num / den
                accessible[k] = 1.0
        return out, accessible

    # initialize a simple tree sequence with two trees
    seq_length=300
    tables = tskit.TableCollection(sequence_length=seq_length)
    sample = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    spans = [(0, 250), (250, 300)]
    for left, right in spans:
        root = tables.nodes.add_row(time=1)
        tables.edges.add_row(left=left, right=right, parent=root, child=sample)
    tables.sort()
    ts = tables.tree_sequence()
    bp = ts.breakpoints(as_array=True).astype(int)
    treespan = np.column_stack([bp[:-1], bp[1:]])
    feature = np.array([1.0, 0.0])
    window_size = 100
    seq_length = 300
    # run both the TRACE and reference windowed aggregation functions
    ncoal_trace, trace_accessible = trace_windowed(
        treespan, feature, window_size, seq_length
    )
    ncoal_reference, reference_accessible = reference_windowed(
        treespan, feature, window_size, seq_length
    )
    # compare the results
    for k in range(len(ncoal_reference)):
        expected_reference = ncoal_reference[k]
        expected_trace = ncoal_trace[k]
        assert np.isclose(expected_reference, expected_trace)
        assert trace_accessible[k] == reference_accessible[k]
    
