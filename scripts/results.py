"""
Compile numerical experiment results into a LaTeX table.
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

# Feature indices
IDX = [
    "SPL_T Mean",
    "SPL_T Std",
    "SPL_S Mean",
    "SPL_S Std",
    "SC_T Mean",
    "SC_T Std",
    "SC_S Mean",
    "SC_S Std",
    "SF_T Mean",
    "SF_T Std",
    "SF_S Mean",
    "SF_S Std",
    "TC Mean",
    "TC Std",
]


def get_folder_metrics(dir):
    # Initialize an empty list to store the dataframes
    dfs = []

    print(f"Processing directory: {dir} ... Found: {Path(dir).exists()}")
    size = Path(dir).parent.name.split("_")[2]
    # Recursively search through folders in the cwd
    for root, dirs, files in os.walk(Path(dir)):
        parts = Path(root).name.split("_")
        if parts[0] == "version":
            version = parts[1]

        for file in files:
            # Check if the file is named "metrics.csv" and has a CSV extension
            if file == "metrics.csv" and file.endswith(".csv"):
                # Construct the file path
                file_path = os.path.join(root, file)

                # Load the CSV file into a dataframe
                df = pd.read_csv(file_path)
                df["version"] = version
                df["size"] = size
                df["preset"] = None

                dfs.append(df)

    # Concatenate all the dataframes into a single dataframe
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df


def update_columns(df):
    new_columns = []
    for c in df.columns:
        if "Loudness" in c[1]:
            feature = "SPL"
        elif "SpectralCentroid" in c[1]:
            feature = "SC"
        elif "SpectralFlatness" in c[1]:
            feature = "SF"

        if "0_" in c[1]:
            feature += "_T"
        else:
            feature += "_S"

        if "TemporalCentroid" in c[1]:
            feature = "TC"

        if "mean" in c[0]:
            feature += " Mean"
        else:
            feature += " Std"

        new_columns.append(feature)
    df.columns = new_columns
    return df


def convert_to_text(df):
    feature = ""
    f_dict = {}
    c_names = df.columns
    for r in df.iterrows():
        if r[0].endswith("Mean"):
            feature = r[0].split(" ")[0]
            values = r[1].values
        elif r[0].endswith("Std"):
            v = []
            for m, s in zip(values, r[1].values):
                if s >= 10:
                    std = f"{s:.0f}"
                else:
                    std = f"{s:.1f}"

                if m >= 10:
                    v.append(f"${m:.1f} \pm {std}$")  # noqa W605
                else:
                    v.append(f"${m:.3f} \pm {std}$")  # noqa W605

            feature = f"${feature}$"
            f_dict[feature] = v

    df1 = pd.DataFrame(f_dict).T
    df1.columns = c_names
    df1 = df1[
        [
            "preset",
            "direct",
            "linear",
            "linear2048",
            "mlp",
            "mlp2048",
            "mlplrg",
            "mlplrg2048",
        ]
    ]
    return df1


def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", help="Input directory", type=str)
    args = parser.parse_args(arguments)

    # Load in all the log files and combine them into a single dataframe
    root = args.input
    df_linear = get_folder_metrics(f"{root}/test_logs_linear/lightning_logs")
    df_linear_2048 = get_folder_metrics(f"{root}/test_logs_linear2048/lightning_logs")

    df_mlp = get_folder_metrics(f"{root}/test_logs_mlp/lightning_logs")
    df_mlp_2048 = get_folder_metrics(f"{root}/test_logs_mlp2048/lightning_logs")

    df_mlp_lrg = get_folder_metrics(f"{root}/test_logs_mlplrg/lightning_logs")
    df_mlp_lrg_2048 = get_folder_metrics(f"{root}/test_logs_mlplrg2048/lightning_logs")

    df_direct = get_folder_metrics(f"{root}/test_logs_direct_opt/lightning_logs")
    combined_df = pd.concat(
        [
            df_linear,
            df_linear_2048,
            df_mlp,
            df_mlp_2048,
            df_direct,
            df_mlp_lrg,
            df_mlp_lrg_2048,
        ],
        ignore_index=True,
    )

    # Convert to a pivot table for easier processing
    columns = [
        c
        for c in combined_df.columns
        if c not in ["epoch", "step", "test/loss", "version", "size", "preset"]
    ]
    columns = [c for c in columns if "pre" not in c]
    df = pd.pivot_table(
        combined_df, values=columns, index=["size"], aggfunc=["mean", "std"]
    )
    df = update_columns(df)
    df = df.T

    # Repeat for the preset
    columns = [
        c
        for c in combined_df.columns
        if c not in ["epoch", "step", "test/loss", "version", "size", "preset"]
    ]
    columns = [c for c in columns if "pre" in c]
    df_pre = pd.pivot_table(
        combined_df, values=columns, index=["size"], aggfunc=["mean", "std"]
    )
    df_pre = update_columns(df_pre)
    df["preset"] = df_pre.T["linear"].values

    df = df.reindex(IDX)
    df = convert_to_text(df)

    # Save the dataframe to a LaTeX table
    df.to_latex("table.tex", escape=False)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
