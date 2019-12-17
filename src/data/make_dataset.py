# src/data/make_dataset.py

from pathlib import Path
from subprocess import run
import sys


def main(phx_project_dpath: Path, phx_output_dpaths_fname: str) -> None:
    """
    Create list of PoolHapX output directories from given PoolHapX project path.

    Parameters
    ----------
    phx_project_dpath : pathlib.Path
        PoolHapX project directory.

    phx_output_dpaths_fname : str
        File path for text file containing PoolHapX output directory paths.
    """
    phx_output_dpaths: Path = Path(
        Path(__file__).absolute().parents[2], "data", "raw", phx_output_dpaths_fname
    )
    run(
        "cp {} {}".format(
            Path(phx_project_dpath, "param_sim", "phx_params.txt"), phx_output_dpaths.parent
        ),
        shell=True,
    )

    with phx_output_dpaths.open("w") as output:
        directory: Path
        for directory in Path(phx_project_dpath, "output").iterdir():
            if directory.is_dir():
                output.write(str(directory.absolute()) + "\n")


if __name__ == "__main__":
    phx_project_dpath: Path = Path(sys.argv[1]).absolute()
    phx_output_dpaths_fname: str = sys.argv[2]

    main(phx_project_dpath=phx_project_dpath, phx_output_dpaths_fname=phx_output_dpaths_fname)
