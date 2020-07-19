"""
Command line utilities for conversion.
"""
import argparse
import os
import signal

from dask.diagnostics import ProgressBar

from . import read_plink


def set_sigpipe_handler():
    if os.name == "posix":
        # Set signal handler for SIGPIPE to quietly kill the program.
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)


def run_plink2sg(args):
    # TODO add args for bim and fam seps
    # TODO add logging and verbosity so we can write out info messages
    # about what we're doing here.
    ds = read_plink(args.plink_path, bim_sep=" ", fam_sep=" ")

    # TODO add a command line options for this progess bar.
    with ProgressBar():

        # TODO Do the dask compute manually so that we can have
        # more influence over the number of threads used.

        # TODO add options to give more control over the zarr
        # format used. We probably want to have the option of
        # writing to a ZipFile anyway, so that we don't have
        # gazillions of files.

        # TODO catch keyboard interrups here, clean up the
        # zarr file, and write a meaningful error message.
        ds.to_zarr(args.sgkit_path, mode="w")
    # TODO write out an INFO summary of the size of the
    # input and output


def add_plink2sg_arguments(parser):

    # TODO we don't seem to have an __version__ defined yet.
    # parser.add_argument(
    #     "-V",
    #     "--version",
    #     action="version",
    #     version=f"%(prog)s {sgkit_plink.__version__}",
    # )

    parser.add_argument("plink_path", help="The plink dataset to read")
    parser.add_argument("sgkit_path", help="The path to write the converted dataset to")


def get_plink2sg_parser(parser=None):

    parser = argparse.ArgumentParser(description="Convert plink files to sgkit format")
    add_plink2sg_arguments(parser)

    return parser


def get_sgkit_plink_parser():
    top_parser = argparse.ArgumentParser(
        description=("Utilities for converting data from plink to sgkit and vice versa")
    )
    subparsers = top_parser.add_subparsers(dest="subcommand")
    subparsers.required = True
    parser = subparsers.add_parser("plink2sg")
    parser.set_defaults(runner=run_plink2sg)
    add_plink2sg_arguments(parser)
    # TODO add sg2plink also
    return top_parser


def plink2sg_main(arg_list=None):
    """
    Top-level hook for the plink2sg console script.
    """
    parser = get_plink2sg_parser()
    args = parser.parse_args(arg_list)
    args.runner(args)


def sgkit_plink_main(arg_list=None):
    """
    Top-level hook called when running python -m sgkit_plink. Just
    exists to call plink2sg or sg2plink as subcommands.
    """
    parser = get_sgkit_plink_parser()
    set_sigpipe_handler()
    args = parser.parse_args(arg_list)
    args.runner(args)
