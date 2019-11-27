# src/features/build_features.py

import csv
from pathlib import Path
import sys
from subprocess import run

import pandas as pd


def _loop_dirs(dir_path: pd.Series, compiled_metrics: pd.DataFrame) -> None:
    """
    Extract MCC and JSD values using shell commands and update compiled metrics dataframe.

    Parameters
    ----------
    dir_path : pd.Series
        A row within the simulation output directory paths dataframe.

    compiled_metrics : pd.DataFrame
        The compiled metrics dataframe to be updated.
    """
    idx: str = dir_path.name
    sim: str = dir_path[0].split("/")[-1]

    mcc: str = run(
        "tail -1 {}/MCC.result".format(dir_path[0]), shell=True, capture_output=True
    ).stdout.decode()[0:-1].split("\t")[1]

    jsd: str = run(
        "tail -1 {}/JSD.result".format(dir_path[0]), shell=True, capture_output=True
    ).stdout.decode()[0:-1].split("\t")[1]

    compiled_metrics.loc[idx] = [sim, float(mcc), float(jsd)]  # add new row


def _compile_metrics(output_dirs: pd.DataFrame, project_data_dir: Path) -> pd.DataFrame:
    """
    Given a dataframe containing all simulation output directories, compile the MCC and JSD metrics
    into a single dataframe and write to a CSV file in data/interim.

    Parameters
    ----------
    output_dirs : pd.DataFrame
        Dataframe containing all simulation output directory paths.

    project_data_dir : pathlib.Path
        The directory path for the data directory in this Cookiecutter project.

    Returns
    -------
    pd.DataFrame
        Dataframe containing MCC and JSD values for all simulations.
    """
    compiled_metrics: pd.DataFrame = pd.DataFrame({"sim": [], "mcc": [], "jsd": []})
    unused: pd.DataFrame = output_dirs.apply(_loop_dirs, axis=1, compiled_metrics=compiled_metrics)
    del unused  # delete unused pandas apply stdout

    pd.to_csv(str(Path(project_data_dir, "interim", "phx_compiled_metrics.csv")), index=False)
    return compiled_metrics


def _process_metrics(compiled_metrics: pd.DataFrame, project_data_dir: Path) -> None:
    """
    Given a dataframe containing all compiled MCC and JSD metrics, subtract JSD from MCC and write
    to a CSV file in data/processed.

    Parameters
    ----------
    compiled_metrics : pd.DataFrame
        Dataframe containing the compiled MCC and JSD metrics.

    project_data_dir : pathlib.Path
        The directory for the data directory in this Cookiecutter project.
    """
    processed_metrics: pd.DataFrame = compiled_metrics
    processed_metrics["mcc_minus_jsd"] = processed_metrics["mcc"] - processed_metrics["jsd"]
    processed_metrics.drop(["mcc", "jsd"], axis=1, inplace=True)

    pd.to_csv(str(Path(project_data_dir, "processed", "phx_processed_metrics.csv")), index=False)


def main(phx_output_dpaths_file: Path) -> None:
    """
    Given a text file containing all simulation output directory paths, compile and process MCC and
    JSD metrics and store in data/interm and data/processed.

    Parameters
    ----------
    output_dir_paths_file : pathlib.Path
        Text file containing paths to each simulation output directory.
    """
    phx_output_dpaths: pd.DataFrame = pd.read_csv(phx_output_dpaths_file, header=None)
    project_data_dpath: Path = phx_output_dpaths_file.cwd().parent
    compiled_metrics: pd.DataFrame = _compile_metrics(phx_output_dpaths, project_data_dpath)
    _process_metrics(compiled_metrics, project_data_dpath)  # save in data/processed/


if __name__ == "__main__":
    phx_output_dpaths_filename: Path = Path(
        Path(__file__).cwd().parents[1], "data", "raw", sys.argv[1]  # in data/raw/
    )

    main(phx_output_dpaths_filename)
