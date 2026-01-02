"""CLI for ghosthmm."""
import logging
import sys
from trace import GhostProductHMM

import click
import numpy as np
import pandas as pd

# Setup the logging configuration for the CLI
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


@click.command()
@click.option(
    "--input",
    "-i",
    required=True,
    type=click.Path(exists=True),
    help="Input data in tskit format.",
)
@click.option(
    "--time",
    "-t",
    required=True,
    type=float,
    default=15e3,
    help="Focal time for branch.",
)
@click.option(
    "--samples",
    "-s",
    required=False,
    type=click.Path(exists=True),
    help="List of sampled individuals to run analysis for.",
)
@click.option(
    "--out",
    "-o",
    required=True,
    type=str,
    default="trace",
    help="Output file prefix.",
)
def main(
    input,
    t,
    out="trace",
):
    """TRACE-Inference CLI."""
    logging.info(f"Starting to read input data {input}...")
