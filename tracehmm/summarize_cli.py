"""CLI for TRACE."""
import logging
import sys

import click
import numpy as np
import pandas as pd

from tracehmm import TRACE, OutputUtils

# Setup the logging configuration for the CLI
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


@click.command()
@click.option(
    "--file",
    help="Posterior probability file from trace-infer, end with .xss.npz. Multiple files are allowed, separated by comma",
    type=str,
    required=True,
)

@click.option(
    "--chrom",
    help="Chromosome ID used in the output file",
    type=str,
    default=None,
)
@click.option(
    "--posterior-threshold",
    "-p",
    help="posterior probability threshold for calling introgression",
    type=float,
    default=0.9,
)
@click.option(
    "--physical-length-threshold",
    help="physical length threshold for calling introgression, in bp",
    type=int,
    default=50000,
)
@click.option(
    "--genetic-distance-threshold",
    help="genetic distance threshold for calling introgression, in cM",
    type=float,
    default=0.05,
)
@click.option(
    "--remove-margin",
    help="remove margin from start and end in states, in kbp",
    type=float,
    default=0,
)
@click.option(
    "--out",
    help="prefix for output file, output file will be named as [out].summary.txt",
    type=str,
    required=True,
)
def main(
    file=
    out="trace",
):
    """TRACE-Inference CLI."""
    logging.info(f"Starting TRACE in mode {mode}...")
    if mode == "extract":
        pass
    if mode == "infer":
        pass
    if mode == "summarize":
        pass
