# src/data/make_dataset.py

from pathlib import Path
from subprocess import run
import sys


def main(phx_project_dpath: Path, phx_output_dpaths_fname: str) -> None:
    """
    """
    phx_output_dpaths: Path = Path(
        Path(__file__).cwd().parents[1], "data", "raw", phx_output_dpaths_fname
    )
    run(
        "cp {} {}".format(
            Path(phx_project_dpath, "param_sim", "phx_params.txt"), phx_output_dpaths.cwd()
        ),
        shell=True,
    )

    with phx_output_dpaths.open("w") as output:
        directory: Path
        for directory in Path(phx_project_dpath, "output").iterdir():
            if directory.is_dir():
                output.write(str(directory.absolute()))


if __name__ == "__main__":
    phx_project_dpath: Path = Path(sys.argv[1])
    phx_output_dpaths_fname: str = sys.argv[2]

    main(phx_project_dpath, phx_output_dpaths_fname)
