import os

import json
import pandas as pd
import dash
from dash import dcc, html, Input, Output
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

    print(f"Processed {len(all_results)} benchmark JSON files.")
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


BENCHMARK_DIR = os.path.join(os.path.dirname(__file__), "benchmark")
subdirs = [
    d
    for d in os.listdir(BENCHMARK_DIR)
    if os.path.isdir(os.path.join(BENCHMARK_DIR, d))
]

# Create Dash app
app = dash.Dash(__name__)

app.layout = html.Div(
    [
        html.H1("Benchmark Results Visualization"),
        html.Div(
            [
                html.Label("Select Benchmark Directory:"),
                dcc.Dropdown(
                    id="directory-dropdown",
                    options=[{"label": d, "value": d} for d in subdirs],
                    value="",
                ),
            ],
            style={"width": "50%", "margin-bottom": "20px"},
        ),
        html.Div(
            [
                html.H2("Execution Time"),
                dcc.Graph(id="execution-time-graph"),
            ],
            style={"width": "100%", "aspectRatio": "16/9"},
        ),
        html.Div(
            [
                html.H2("Numerical Accuracy"),
                dcc.Graph(id="residual-graph"),
            ],
            style={"width": "100%", "aspectRatio": "16/9"},
        ),
    ]
)


@app.callback(
    [Output("execution-time-graph", "figure"), Output("residual-graph", "figure")],
    [Input("directory-dropdown", "value")],
)
def update_graphs(selected_dir):
    if selected_dir is None or selected_dir == "":
        return {}, {}

    df = load_benchmark_results(os.path.join(BENCHMARK_DIR, selected_dir))

    execution_time_traces = []
    residual_traces = []

    if df.shape[0] == 0:
        # Directory exists but no result data --> skip
        return {}, {}

    solver_types = sorted(df["solver_type"].unique())

    for solver_type in solver_types:
        method_data = df[df["solver_type"] == solver_type]
        execution_time_traces.append(
            go.Scatter(
                x=method_data["batch_size"],
                y=method_data["time_median"],
                mode="lines+markers",
                name=solver_type,
            )
        )
        residual_traces.append(
            go.Scatter(
                x=method_data["batch_size"],
                y=method_data["residuals_l2_avg"],
                mode="lines+markers",
                name=solver_type,
            )
        )

    execution_time_fig = {
        "data": execution_time_traces,
        "layout": go.Layout(
            xaxis={
                "title": "Batch Size",
                "type": "log",
                "tickvals": sorted(df["batch_size"].unique()),
            },
            yaxis={"title": "Median Time (s)", "type": "log"},
            legend={"title": "Solver"},
            title="Execution time (1 analyze + 1 factorize + 1 solve)",
            autosize=True,
        ),
    }

    residual_fig = {
        "data": residual_traces,
        "layout": go.Layout(
            xaxis={
                "title": "Batch Size",
                "type": "log",
                "tickvals": sorted(df["batch_size"].unique()),
            },
            yaxis={"title": "Average l2 norm of residuals", "type": "log"},
            legend={"title": "Solver"},
            title="Numerical accuracy",
            autosize=True,
        ),
    }
    return execution_time_fig, residual_fig


if __name__ == "__main__":
    app.run(debug=True)
