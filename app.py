import os

import json
import pandas as pd
import dash
from dash import dcc, html
import plotly.graph_objs as go


# Process JSON-format data to CSV format
# (this function is only invoked as a last resort, if CSV files are not available)
def process_benchmark_results(results_dir):
    """Process all JSON-format benchmark results in given directory and save them as a CSV file."""
    all_files = os.listdir(results_dir)
    all_results = []
    for fname in all_files:
        if fname.endswith(".bench.json"):
            with open(os.path.join(results_dir, fname), "r") as f:
                data = json.load(f)
            all_results.append(data)

    df = pd.DataFrame(all_results)
    df.sort_values(["solver_type", "batch_size"], inplace=True)
    df.to_csv(os.path.join(results_dir, "benchmark_results.csv"), index=False)

    return None

def load_benchmark_results(results_dir, process_if_missing=True):
    """Load benchmark results from CSV file.
    
    If CSV file is not found, optionally process JSON files to generate it.
    The returned DataFrame is sorted by `solver_type` and `batch_size`.
    """
    if not os.path.exists(os.path.join(results_dir, "benchmark_results.csv")):
        if process_if_missing:
            print("CSV benchmark results not found, processing JSON files...")
            process_benchmark_results(results_dir)
        else:
            print("Error: CSV benchmark results not found, and processing is disabled.")
            return None
    
    # If no CSV file still exists, error
    if not os.path.exists(os.path.join(results_dir, "benchmark_results.csv")):
        print("Error: could not find or generate CSV benchmark results.")
        return None

    df = pd.read_csv(os.path.join(results_dir, "benchmark_results.csv"))
    df.sort_values(["solver_type", "batch_size"], inplace=True)
    return df


df = load_benchmark_results("benchmark/DiffOpt", process_if_missing=True)

# Define the methods to be plotted
methods = sorted(df.solver_type.unique())

# Create traces for execution time graph
execution_time_traces = []
for method in methods:
    method_data = df[df["solver_type"] == method]
    execution_time_traces.append(
        go.Scatter(
            x=method_data["batch_size"],
            y=method_data["time_median"],
            mode="lines+markers",
            name=method,
        )
    )

# Create traces for residual graph
residual_traces = []
for method in methods:
    method_data = df[df["solver_type"] == method]
    residual_traces.append(
        go.Scatter(
            x=method_data["batch_size"],
            y=method_data["residuals_l2_avg"],
            mode="lines+markers",
            name=method,
        )
    )

# Create Dash app
app = dash.Dash(__name__)

app.layout = html.Div(
    [
        html.H1("Benchmark Results Visualization"),
        html.Div(
            [
                html.H2("Execution Time vs Batch Size"),
                dcc.Graph(
                    figure={
                        "data": execution_time_traces,
                        "layout": go.Layout(
                            xaxis={
                                "title": "Batch Size",
                                "type": "log",
                                "tickvals": [8, 16, 32, 64, 128, 256, 1024],
                            },
                            yaxis={
                                "title": "Median Time (s)",
                                "type": "log",
                            },
                            legend={"title": "Solver"},
                            title="Execution time (1 analyze + 1 factorize + 1 solve)",
                        ),
                    }
                ),
            ]
        ),
        html.Div(
            [
                html.H2("Residual vs Batch Size"),
                dcc.Graph(
                    figure={
                        "data": residual_traces,
                        "layout": go.Layout(
                            xaxis={
                                "title": "Batch Size",
                                "type": "log",
                                "tickvals": [8, 16, 32, 64, 128, 256, 1024],
                            },
                            yaxis={"title": "Residual", "type": "log"},
                            legend={"title": "Solver"},
                            title="Numerical accuracy",
                        ),
                    }
                ),
            ]
        ),
    ]
)

if __name__ == "__main__":
    app.run(debug=True)
