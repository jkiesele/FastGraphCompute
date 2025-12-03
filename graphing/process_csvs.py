import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Global configuration for algorithms
ALGORITHMS = {
    'FAISS': {
        'display_name': 'FAISS',
        'color': '#1B9E77',  # Professional green
        'time_column': 'faiss_time',
        'marker_symbol': 'circle'
    },
    'SCANN': {
        'display_name': 'ScaNN',
        'color': '#D95F02',  # Professional orange
        'time_column': 'scann_time',
        'marker_symbol': 'triangle-up'
    },
    'HNSWLIB': {
        'display_name': 'HNSWLIB',
        'color': '#2E86AB',  # Professional blue
        'time_column': 'hnswlib_time',
        'marker_symbol': 'diamond'
    },
    'ANNOY': {
        'display_name': 'Annoy',
        'color': '#A23B72',  # Professional magenta
        'time_column': 'annoy_time',
        'marker_symbol': 'square'
    },
    'GGNN': {
        'display_name': 'GGNN',
        'color': '#7570B3',  # Professional purple
        'time_column': 'ggnn_time',
        'marker_symbol': 'cross'
    },
    'FGC': {
        'display_name': 'FGC',
        'color': '#F18F01',  # Professional yellow-orange
        'time_column': 'fgc_time',
        'marker_symbol': 'star'
    }
}

MAX_DATASET_SIZE = 5_000_000  # Cap all analyses at 5M data points

# Processing mode: Choose one of the following options
# - "process": Process CSV files from directories and create _cleaned.csv files
# - "use_existing": Use existing _cleaned.csv files (skip processing)
# - "merge": Merge all _cleaned.csv files into master_data.csv with all algorithms combined
PROCESSING_MODE = "process"  # Options: "process", "use_existing", "merge"


def combine_algorithm_files(algorithm_dir: str, algorithm_name: str, time_column: str) -> pd.DataFrame:
    """
    Combine all CSV files from an algorithm's data directory.

    Args:
        algorithm_dir: Path to the directory containing algorithm CSV files.
        algorithm_name: Name of the algorithm (for display purposes).
        time_column: Name of the time column for this algorithm (e.g., 'scann_time').

    Returns:
        Combined DataFrame with standardized columns.
    """
    if not os.path.exists(algorithm_dir):
        print(f"{algorithm_name} directory not found: {algorithm_dir}")
        return pd.DataFrame()

    print(f"Combining {algorithm_name} files from {algorithm_dir}...")
    all_data = []

    # Get all CSV files in the directory
    csv_files = [f for f in os.listdir(algorithm_dir) if f.endswith(
        '.csv') and not f.startswith('.')]
    print(f"Found {len(csv_files)} {algorithm_name} CSV files to process")

    for filename in csv_files:
        file_path = os.path.join(algorithm_dir, filename)
        try:
            df = pd.read_csv(file_path)
            print(f"  Processing {filename}: {len(df)} records")

            # Standardize column names - handle 'dims' -> 'dimension'
            if 'dims' in df.columns and 'dimension' not in df.columns:
                df = df.rename(columns={'dims': 'dimension'})

            # Standardize columns based on file type
            if 'size' in df.columns and 'fixed_dimension' in df.columns:
                # Size-varying files: size is variable, fixed_dimension is constant
                df = df.rename(columns={'fixed_dimension': 'dimension'})
            elif 'dimension' in df.columns and 'fixed_size' in df.columns:
                # Dimension-varying files: dimension is variable, fixed_size is constant
                df = df.rename(columns={'fixed_size': 'size'})

            # Ensure all required columns are present
            required_cols = ['size', 'k', 'dimension',
                             time_column, 'fgc_time', 'count']
            if not all(col in df.columns for col in required_cols):
                print(
                    f"  Warning: Missing required columns in {filename}, skipping")
                continue

            # Select only the required columns
            df = df[required_cols]
            all_data.append(df)

        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            continue

    if not all_data:
        print(f"No valid {algorithm_name} data found!")
        return pd.DataFrame()

    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Combined {algorithm_name} data: {len(combined_df)} total records")

    return combined_df


def process_algorithm_data(algorithm_name: str, algorithm_dir: str, output_path: str, time_column: str):
    """
    Process algorithm data by combining CSV files and applying weighted averages.

    Args:
        algorithm_name: Name of the algorithm (for display purposes).
        algorithm_dir: Path to the directory containing algorithm CSV files.
        output_path: Path to save the cleaned combined CSV file.
        time_column: Name of the time column for this algorithm (e.g., 'annoy_time').
    """
    print(f"\nProcessing {algorithm_name} data...")

    # Combine all CSV files from the algorithm directory
    combined_df = combine_algorithm_files(
        algorithm_dir, algorithm_name, time_column)

    if combined_df.empty:
        print(f"No {algorithm_name} data to process!")
        return

    # Define processing parameters
    time_cols = [time_column, 'fgc_time']
    key_cols = ['size', 'k', 'dimension']

    # Ensure all necessary columns are present
    required_cols = key_cols + time_cols + ['count']
    if not all(col in combined_df.columns for col in required_cols):
        print(
            f"Missing required columns in combined {algorithm_name} data. Skipping.")
        return

    print(
        f"Processing combined {algorithm_name} data: {len(combined_df)} records")

    # Define aggregation logic for weighted averages
    agg_funs = {}
    for col in time_cols:
        agg_funs[col] = lambda x: np.average(
            x, weights=combined_df.loc[x.index, 'count'])

    agg_funs['count'] = 'sum'

    # Group and aggregate
    df_cleaned = combined_df.groupby(key_cols).agg(agg_funs).reset_index()

    # Save cleaned data
    df_cleaned.to_csv(output_path, index=False)
    print(f"Saved cleaned {algorithm_name} data to {output_path}")


def load_and_prepare_data() -> pd.DataFrame:
    """Load and prepare unified algorithm performance data."""
    print("Loading performance data...")

    # Check if master_data.csv exists (from merge mode)
    if os.path.exists('master_data.csv'):
        print("Found master_data.csv, loading directly...")
        unified_data = pd.read_csv('master_data.csv')

        # Standardize column names
        if 'dims' in unified_data.columns and 'dimension' not in unified_data.columns:
            unified_data = unified_data.rename(columns={'dims': 'dimension'})

        # Count records per algorithm
        for alg in ['FAISS', 'SCANN', 'HNSWLIB', 'ANNOY', 'GGNN']:
            alg_time_col = ALGORITHMS[alg]['time_column']
            if alg_time_col in unified_data.columns:
                count = unified_data[alg_time_col].notna().sum()
                print(f"Loaded {alg}: {count} records")

        print(
            f"Loaded unified data from master_data.csv: {len(unified_data)} records")
        return unified_data

    # Otherwise, load individual cleaned files and merge them
    data_files = {
        'FAISS': 'faiss_data_cleaned.csv',
        'SCANN': 'scann_data_cleaned.csv',
        'HNSWLIB': 'hnswlib_data_cleaned.csv',
        'ANNOY': 'annoy_data_cleaned.csv',
        'GGNN': 'ggnn_data_cleaned.csv'
    }

    datasets = {}
    for alg, filename in data_files.items():
        if os.path.exists(filename):
            datasets[alg] = pd.read_csv(filename)
            print(f"Loaded {alg}: {len(datasets[alg])} records")
        else:
            print(f"Warning: {filename} not found, skipping {alg}")

    if not datasets:
        print("No cleaned data files found!")
        return pd.DataFrame()

    # Start with the first available dataset as base
    base_alg = list(datasets.keys())[0]
    unified_data = datasets[base_alg].copy()

    # Standardize column names - ensure 'dimension' column exists
    if 'dims' in unified_data.columns and 'dimension' not in unified_data.columns:
        unified_data = unified_data.rename(columns={'dims': 'dimension'})

    # Merge other algorithms
    for alg, df in datasets.items():
        if alg == base_alg:
            continue

        # Standardize column names
        if 'dims' in df.columns and 'dimension' not in df.columns:
            df = df.rename(columns={'dims': 'dimension'})

        # Get the time columns for this algorithm
        alg_time_col = ALGORITHMS[alg]['time_column']
        merge_cols = ['size', 'k', 'dimension', alg_time_col, 'fgc_time']

        # Only keep columns that exist
        available_cols = [col for col in merge_cols if col in df.columns]

        if len(available_cols) >= 3:  # At least size, k, dimension
            unified_data = pd.merge(
                unified_data,
                df[available_cols],
                on=['size', 'k', 'dimension'],
                how='outer',
                suffixes=('', f'_{alg.lower()}')
            )

    print(f"Loaded unified data: {len(unified_data)} records")
    return unified_data


def plot_fgc_speedup_analysis(data: pd.DataFrame, analysis_type: str, **kwargs) -> go.Figure:
    """Create FGC speedup analysis plot with all algorithms."""

    y_axis_cap = kwargs.get('y_axis_cap', None)
    custom_title = kwargs.get('custom_title', None)
    log_y = kwargs.get('log_y', False)

    if analysis_type == 'dimensions':
        size = kwargs.get('size', 1_000_000)
        k = kwargs.get('k', 40)
        max_dimensions = kwargs.get('max_dimensions', 20)
        title = f"FGC Speedup Analysis: {size//1_000_000}M Points, K={k}"
        if y_axis_cap:
            title += f" (Y-Axis Capped at {y_axis_cap})"

        # Filter data
        filtered_data = data[
            (data['size'] == size) &
            (data['k'] == k) &
            (data['dimension'] <= max_dimensions)
        ].copy().sort_values('dimension')

        x_col = 'dimension'
        x_title = "Number of Dimensions (d)"
        x_range = [1, max_dimensions]

    else:  # sizes
        dimension = kwargs.get('dimension', 3)
        k = kwargs.get('k', 40)
        title = f"FGC Speedup Analysis: D={dimension}, K={k}, Varying Sizes"
        if y_axis_cap:
            title += f" (Y-Axis Capped at {y_axis_cap})"

        # Filter data
        filtered_data = data[
            (data['dimension'] == dimension) &
            (data['k'] == k) &
            (data['size'] <= MAX_DATASET_SIZE)
        ].copy().sort_values('size')

        # Filter to only show specific sizes: 0, 100k, 500k, 1M, then every 500k
        allowed_sizes = [0, 100_000, 500_000, 1_000_000]
        # Add every 500k after 1M up to MAX_DATASET_SIZE
        current_size = 1_500_000
        while current_size <= MAX_DATASET_SIZE:
            allowed_sizes.append(current_size)
            current_size += 500_000

        filtered_data = filtered_data[filtered_data['size'].isin(
            allowed_sizes)].copy()

        x_col = 'size'
        x_title = "Dataset Size"
        x_range = [0, 5_000_000]

    # Create figure
    fig = go.Figure()

    # Plot all algorithms
    for algorithm in ['FAISS', 'SCANN', 'HNSWLIB', 'ANNOY', 'GGNN']:
        if algorithm not in ALGORITHMS:
            continue

        alg_info = ALGORITHMS[algorithm]
        time_col = alg_info['time_column']

        # Determine which fgc_time column to use
        # Priority: algorithm-specific fgc_time > base fgc_time
        # This handles both merged master_data.csv (single fgc_time) and individual files (algorithm-specific)
        if algorithm == 'FAISS':
            fgc_col = 'fgc_time'
        else:
            fgc_col = f'fgc_time_{algorithm.lower()}'
            # Fall back to base fgc_time if algorithm-specific doesn't exist
            # (This happens when using master_data.csv which has a single merged fgc_time)
            if fgc_col not in filtered_data.columns:
                fgc_col = 'fgc_time'

        if time_col not in filtered_data.columns or fgc_col not in filtered_data.columns:
            continue

        # Calculate speedup
        valid_data = filtered_data[
            (filtered_data[time_col].notna()) &
            (filtered_data[fgc_col].notna()) &
            (filtered_data[time_col] > 0) &
            (filtered_data[fgc_col] > 0)
        ].copy()

        if valid_data.empty:
            continue

        speedup = valid_data[time_col] / valid_data[fgc_col]

        # Add trace
        fig.add_trace(
            go.Scatter(
                x=valid_data[x_col],
                y=speedup,
                mode='lines+markers',
                name=alg_info['display_name'],
                line=dict(color=alg_info['color'], width=3),
                marker=dict(size=8, symbol=alg_info['marker_symbol']),
                showlegend=True,
                hovertemplate=(
                    f"<b>{alg_info['display_name']}</b><br>"
                    f"{x_title}: %{{x}}<br>"
                    "Speedup: %{{y:.2f}}×<br>"
                    "<extra></extra>"
                )
            )
        )

    # Add reference line at y=1
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=[1, 1],
            mode='lines',
            name='No Speedup',
            line=dict(color='gray', dash='dash', width=2),
            showlegend=True,
            hovertemplate="No speedup reference<extra></extra>"
        )
    )

    # Update layout
    fig.update_xaxes(
        range=x_range,
        gridcolor='lightgray',
        title=x_title,
        title_font_size=18,
        tickfont=dict(size=16)
    )

    if analysis_type == 'sizes':
        # Set tick marks to show only every 1M: 0, 1M, 2M, 3M, 4M, 5M
        tick_vals = [0]
        tick_texts = ['0']
        current_size = 1_000_000
        while current_size <= MAX_DATASET_SIZE:
            tick_vals.append(current_size)
            tick_texts.append(f'{current_size//1_000_000}M')
            current_size += 1_000_000

        fig.update_xaxes(
            tickmode='array',
            tickvals=tick_vals,
            ticktext=tick_texts
        )

    y_axis_config = {
        'gridcolor': 'lightgray',
        'title': "FGC Speedup Factor",
        'title_font_size': 18,
        'tickfont': dict(size=16)
    }
    if y_axis_cap:
        if log_y:
            # Start at 1 (log10(1)=0) to avoid markers between 0 and 1
            y_axis_config['range'] = [0, np.log10(y_axis_cap)]
        else:
            y_axis_config['range'] = [0, y_axis_cap]

    if log_y:
        y_axis_config['type'] = 'log'
        y_axis_config['dtick'] = 1  # Major ticks every power of 10
        # Disable minor ticks/grid to ensure visually uniform spacing
        y_axis_config['minor'] = dict(
            showgrid=False,
            ticklen=0
        )
        y_axis_config['tickformat'] = '.0f'  # Display as 1, 10, 100

    fig.update_yaxes(**y_axis_config)

    if custom_title:
        title = custom_title

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font_size=21,
            font_family="Arial"
        ),
        height=500,
        width=800 if analysis_type == 'dimensions' else 1200,
        template='plotly_white',
        font=dict(family="Arial", size=16),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="right",
            x=0.98,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="lightgray",
            borderwidth=1
        ),
        margin=dict(t=80, b=60, l=80, r=80)
    )

    return fig


def plot_side_by_side_with_zoom(data: pd.DataFrame, analysis_type: str, **kwargs) -> go.Figure:
    """
    Create side-by-side plot with standard and zoomed (y-axis capped) views.

    Args:
        data: Unified DataFrame with all algorithm data
        analysis_type: 'dimensions' or 'sizes'
        **kwargs: Additional parameters including y_axis_cap for zoomed view
    """
    y_axis_cap = kwargs.pop('y_axis_cap', 50)

    if analysis_type == 'dimensions':
        dimension = kwargs.get('dimension', None)
        size = kwargs.get('size', 1_000_000)
        k = kwargs.get('k', 40)
        max_dimensions = kwargs.get('max_dimensions', 15)

        subtitle_left = f"Standard View (d ≤ {max_dimensions})"
        subtitle_right = f"Zoomed View (Y-Axis Capped at {y_axis_cap})"
        main_title = f"FGC Dimensional Scaling Analysis ({size//1_000_000}M Vectors, K={k})"
    else:  # sizes
        dimension = kwargs.get('dimension', 3)
        k = kwargs.get('k', 40)
        subtitle_left = f"Standard View"
        subtitle_right = f"Zoomed View (Y-Axis Capped at {y_axis_cap})"
        main_title = f"FGC Speedup Analysis: D={dimension}, K={k}, Varying Sizes"

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(subtitle_left, subtitle_right),
        horizontal_spacing=0.08
    )
    # Update subplot title font size
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=16)

    # Generate standard view
    fig_normal = plot_fgc_speedup_analysis(
        data, analysis_type, custom_title=" ", **kwargs)

    # Generate zoomed view
    fig_zoomed = plot_fgc_speedup_analysis(
        data, analysis_type, custom_title=" ", y_axis_cap=y_axis_cap, **kwargs)

    # Add traces from normal plot to first subplot
    for trace in fig_normal.data:
        fig.add_trace(trace, row=1, col=1)

    # Add traces from zoomed plot to second subplot (no legend to avoid duplicates)
    for trace in fig_zoomed.data:
        trace.showlegend = False
        fig.add_trace(trace, row=1, col=2)

    # Update axes
    if analysis_type == 'dimensions':
        x_title = "Number of Dimensions (d)"
    else:
        x_title = "Dataset Size"

    fig.update_xaxes(title_text=x_title, title_font_size=18,
                     tickfont=dict(size=16), row=1, col=1)
    fig.update_xaxes(title_text=x_title, title_font_size=18,
                     tickfont=dict(size=16), row=1, col=2)
    fig.update_yaxes(title_text="FGC Speedup Factor",
                     title_font_size=18, tickfont=dict(size=16), row=1, col=1)
    fig.update_yaxes(title_text="FGC Speedup Factor", title_font_size=18, tickfont=dict(size=16),
                     range=[0, y_axis_cap], row=1, col=2)

    if analysis_type == 'sizes':
        # Set tick marks to show only every 1M: 0, 1M, 2M, 3M, 4M, 5M
        tick_vals = [0]
        tick_texts = ['0']
        current_size = 1_000_000
        while current_size <= MAX_DATASET_SIZE:
            tick_vals.append(current_size)
            tick_texts.append(f'{current_size//1_000_000}M')
            current_size += 1_000_000

        for col in [1, 2]:
            fig.update_xaxes(
                tickmode='array',
                tickvals=tick_vals,
                ticktext=tick_texts,
                row=1, col=col
            )

    # Update main layout - position legend on rightmost subplot (col=2)
    fig.update_layout(
        title_text=main_title,
        title_font_size=21,
        height=600,
        width=1400,
        template='plotly_white',
        font=dict(family="Arial", size=16),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="right",
            x=0.98,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="lightgray",
            borderwidth=1
        ),
        margin=dict(t=80, b=60, l=80, r=80)
    )

    return fig


def plot_d3_d5_comparison(data: pd.DataFrame, k: int = 40) -> go.Figure:
    """Create side-by-side plot comparing D=3 and D=5 size iteration analysis."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("D=3", "D=5"),
        horizontal_spacing=0.08
    )
    # Update subplot title font size
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=16)

    # Generate plots for d=3 and d=5
    fig_d3 = plot_fgc_speedup_analysis(
        data, 'sizes', dimension=3, k=k, custom_title=" ")

    fig_d5 = plot_fgc_speedup_analysis(
        data, 'sizes', dimension=5, k=k, custom_title=" ")

    # Add traces from d=3 plot to first subplot (no legend to avoid duplicates)
    for trace in fig_d3.data:
        trace.showlegend = False
        fig.add_trace(trace, row=1, col=1)

    # Add traces from d=5 plot to second subplot (show legend on rightmost subplot)
    for trace in fig_d5.data:
        fig.add_trace(trace, row=1, col=2)

    # Update axes
    x_title = "Dataset Size"

    # Set tick marks to show only every 1M: 0, 1M, 2M, 3M, 4M, 5M
    tick_vals = [0]
    tick_texts = ['0']
    current_size = 1_000_000
    while current_size <= MAX_DATASET_SIZE:
        tick_vals.append(current_size)
        tick_texts.append(f'{current_size//1_000_000}M')
        current_size += 1_000_000

    for col in [1, 2]:
        fig.update_xaxes(
            title_text=x_title,
            title_font_size=18,
            tickfont=dict(size=16),
            tickmode='array',
            tickvals=tick_vals,
            ticktext=tick_texts,
            row=1, col=col
        )
        fig.update_yaxes(
            title_text="FGC Speedup Factor" if col == 1 else "",
            title_font_size=18 if col == 1 else None,
            tickfont=dict(size=16),
            row=1, col=col
        )

    # Update main layout - position legend on rightmost subplot (col=2)
    fig.update_layout(
        title_text=f"FGC Speedup Analysis: D=3 vs D=5 Comparison (K={k}, Varying Sizes)",
        title_font_size=21,
        height=600,
        width=1400,
        template='plotly_white',
        font=dict(family="Arial", size=16),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="right",
            x=0.98,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="lightgray",
            borderwidth=1
        ),
        margin=dict(t=80, b=60, l=80, r=80)
    )

    return fig


def plot_k_comparison_dimensional_analysis(data: pd.DataFrame) -> go.Figure:
    """Create three-panel side-by-side plot for dimensional analysis at 1M points."""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("K=10", "K=40", "K=100"),
        horizontal_spacing=0.08
    )
    # Update subplot title font size
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=16)

    k_values = [10, 40, 100]

    for i, k in enumerate(k_values, 1):
        # Generate individual figure for this k value
        single_fig = plot_fgc_speedup_analysis(
            data, 'dimensions',
            size=1_000_000,
            k=k,
            max_dimensions=10
        )

        # Add traces from this figure to the appropriate subplot
        for j, trace in enumerate(single_fig.data):
            # Only show legend for the first subplot to avoid duplicates
            trace.showlegend = (i == 1)
            fig.add_trace(trace, row=1, col=i)

    # Update axes for all subplots
    for i in range(1, 4):
        fig.update_xaxes(
            title_text="Number of Dimensions (d)",
            title_font_size=18,
            tickfont=dict(size=16),
            range=[2, 10],
            row=1, col=i
        )
        fig.update_yaxes(
            title_text="FGC Speedup Factor" if i == 1 else "",
            title_font_size=18 if i == 1 else None,
            tickfont=dict(size=16),
            row=1, col=i
        )

    # Update main layout - position legend on rightmost subplot (col=3)
    fig.update_layout(
        title_text="FGC Dimensional Scaling Analysis: K Comparison (1M Vectors, d=2-10)",
        title_font_size=21,
        height=600,
        width=1800,
        template='plotly_white',
        font=dict(family="Arial", size=16),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="right",
            x=0.98,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="lightgray",
            borderwidth=1
        ),
        margin=dict(t=80, b=60, l=80, r=80)
    )
    return fig


def save_figure(fig: go.Figure, filename: str, width: int = 1200, height: int = 600):
    """Save figure with professional settings."""
    fig.write_image(filename, width=width, height=height, scale=2)
    print(f"✓ Saved: {filename}")


def merge_all_csv_files(output_path: str = 'master_data.csv'):
    """
    Merge all cleaned CSV files into a single master CSV file.

    For each unique combination of (size, k, dimension):
    - Combines all algorithm times (annoy_time, ggnn_time, scann_time, hnswlib_time, faiss_time)
    - Combines fgc_time values across all algorithms using weighted average (weighted by count)

    Args:
        output_path: Path to save the master CSV file.
    """
    print("\n" + "="*60)
    print("Merging All CSV Files into Master Data")
    print("="*60)

    # Load all cleaned datasets
    data_files = {
        'FAISS': 'faiss_data_cleaned.csv',
        'SCANN': 'scann_data_cleaned.csv',
        'HNSWLIB': 'hnswlib_data_cleaned.csv',
        'ANNOY': 'annoy_data_cleaned.csv',
        'GGNN': 'ggnn_data_cleaned.csv'
    }

    datasets = {}
    for alg, filename in data_files.items():
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            # Standardize column names
            if 'dims' in df.columns and 'dimension' not in df.columns:
                df = df.rename(columns={'dims': 'dimension'})
            datasets[alg] = df
            print(f"Loaded {alg}: {len(df)} records")
        else:
            print(f"Warning: {filename} not found, skipping {alg}")

    if not datasets:
        print("No cleaned data files found!")
        return

    # Collect all unique combinations of (size, k, dimension)
    all_keys = set()
    for df in datasets.values():
        keys = df[['size', 'k', 'dimension']].drop_duplicates()
        for _, row in keys.iterrows():
            all_keys.add((row['size'], row['k'], row['dimension']))

    print(f"\nFound {len(all_keys)} unique (size, k, dimension) combinations")

    # Build master data
    master_rows = []

    for size, k, dimension in sorted(all_keys):
        row_data = {
            'size': size,
            'k': k,
            'dimension': dimension,
            'annoy_time': None,
            'ggnn_time': None,
            'scann_time': None,
            'hnswlib_time': None,
            'faiss_time': None,
            'fgc_time': None,  # Base fgc_time (from FAISS)
            'fgc_time_annoy': None,
            'fgc_time_scann': None,
            'fgc_time_hnswlib': None,
            'fgc_time_ggnn': None,
            'count': 0
        }

        # Collect algorithm times and fgc_time from each dataset
        for alg, df in datasets.items():
            alg_time_col = ALGORITHMS[alg]['time_column']

            # Find matching rows
            matches = df[
                (df['size'] == size) &
                (df['k'] == k) &
                (df['dimension'] == dimension)
            ]

            if len(matches) > 0:
                # Use the first match (should be unique after cleaning)
                match = matches.iloc[0]

                # Store algorithm time
                if alg_time_col in match.index:
                    alg_time_val = match[alg_time_col]
                    if pd.notna(alg_time_val) and alg_time_val > 0:
                        row_data[alg_time_col] = alg_time_val

                # Store algorithm-specific fgc_time
                if 'fgc_time' in match.index:
                    fgc_time_val = match['fgc_time']
                    if pd.notna(fgc_time_val) and fgc_time_val > 0:
                        if alg == 'FAISS':
                            row_data['fgc_time'] = fgc_time_val
                        else:
                            fgc_col_name = f'fgc_time_{alg.lower()}'
                            row_data[fgc_col_name] = fgc_time_val

                        # Also update count (use the count from this algorithm)
                        count_val = match.get('count', 1)
                        if pd.notna(count_val) and count_val > 0:
                            row_data['count'] = max(
                                row_data['count'], count_val)

        master_rows.append(row_data)

    # Create master DataFrame
    master_df = pd.DataFrame(master_rows)

    # Sort by size, k, dimension
    master_df = master_df.sort_values(
        ['size', 'k', 'dimension']).reset_index(drop=True)

    # Save master CSV
    master_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved master data to {output_path}")
    print(f"  Total records: {len(master_df)}")
    print(f"  Columns: {', '.join(master_df.columns)}")
    print("="*60)


def create_plots():
    """Create all required plots after data processing."""
    print("\n" + "="*60)
    print("Creating Performance Analysis Plots")
    print("="*60)

    # Load unified data
    data = load_and_prepare_data()

    if data.empty:
        print("No data available for plotting!")
        return

    # Create plots directory if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # 1. d3: K=40, size iterate (side-by-side with zoom)
    print("\n1. Creating FGC speedup analysis: D=3, K=40, varying sizes (with zoom)...")
    fig1 = plot_side_by_side_with_zoom(
        data, 'sizes', dimension=3, k=40, y_axis_cap=150)
    save_figure(fig1, 'plots/fgc_speedup_d3_all_algorithms.png', 1400, 600)

    # 2. d5: K=40, size iterate (side-by-side with zoom)
    print("\n2. Creating FGC speedup analysis: D=5, K=40, varying sizes (with zoom)...")
    fig2 = plot_side_by_side_with_zoom(
        data, 'sizes', dimension=5, k=40, y_axis_cap=50)
    save_figure(fig2, 'plots/fgc_speedup_d5_all_algorithms.png', 1400, 600)

    # 3. k comparison: 10, 40, 100 @ dimension iteration, 1M size
    print("\n3. Creating K comparison dimensional analysis: K=10,40,100 @ 1M...")
    fig3 = plot_k_comparison_dimensional_analysis(data)
    save_figure(
        fig3, 'plots/fgc_k_comparison_1M_d2-10_all_algorithms.png', 1800, 600)

    # 4. dimensional scaling, 1M, K=40 (side-by-side with zoom)
    print("\n4. Creating dimensional scaling analysis: 1M, K=40 (with zoom)...")
    fig4 = plot_side_by_side_with_zoom(
        data, 'dimensions', size=1_000_000, k=40, max_dimensions=15, y_axis_cap=50)
    save_figure(
        fig4, 'plots/fgc_dimensional_scaling_1M_k40_all_algorithms.png', 1400, 600)

    # 5. d3 vs d5 comparison: K=40, size iteration (side-by-side)
    print("\n5. Creating D=3 vs D=5 comparison: K=40, varying sizes...")
    fig5 = plot_d3_d5_comparison(data, k=40)
    save_figure(
        fig5, 'plots/fgc_speedup_d3_vs_d5_k40_all_algorithms.png', 1400, 600)

    # 6. d3, k=40 with logarithmic y-axis
    print("\n6. Creating FGC speedup analysis: D=3, K=40, varying sizes (logarithmic y-axis)...")
    fig6 = plot_fgc_speedup_analysis(
        data, 'sizes', dimension=3, k=40, custom_title="FastGraph Speedup at K=40, D=3", log_y=True)
    save_figure(fig6, 'plots/fgc_speedup_d3_k40_log_y.png', 1200, 500)

    print(f"\n" + "="*60)
    print("Plot Generation Complete! Generated files:")
    print("• plots/fgc_speedup_d3_all_algorithms.png (side-by-side with zoom)")
    print("• plots/fgc_speedup_d5_all_algorithms.png (side-by-side with zoom)")
    print("• plots/fgc_k_comparison_1M_d2-10_all_algorithms.png")
    print("• plots/fgc_dimensional_scaling_1M_k40_all_algorithms.png (side-by-side with zoom)")
    print("• plots/fgc_speedup_d3_vs_d5_k40_all_algorithms.png (side-by-side comparison)")
    print("• plots/fgc_speedup_d3_k40_log_y.png (logarithmic y-axis)")
    print("="*60)


def main():
    """Main function to process all algorithm CSVs."""

    algorithms_to_process = [
        {
            'name': 'GGNN',
            'directory': 'ggnn-data',
            'time_column': 'ggnn_time',
            'output': 'ggnn_data_cleaned.csv'
        },
        {
            'name': 'Annoy',
            'directory': 'annoy-data',
            'time_column': 'annoy_time',
            'output': 'annoy_data_cleaned.csv'
        },
        {
            'name': 'FAISS',
            'directory': 'faiss-data',
            'time_column': 'faiss_time',
            'output': 'faiss_data_cleaned.csv'
        },
        {
            'name': 'HNSWLIB',
            'directory': 'hnswlib-data',
            'time_column': 'hnswlib_time',
            'output': 'hnswlib_data_cleaned.csv'
        },
        {
            'name': 'SCANN',
            'directory': 'scann-data',
            'time_column': 'scann_time',
            'output': 'scann_data_cleaned.csv'
        },
    ]

    if PROCESSING_MODE == "process":
        print("Starting CSV processing...")

        # Process all algorithms using the combined file approach
        for algo in algorithms_to_process:
            process_algorithm_data(
                algorithm_name=algo['name'],
                algorithm_dir=algo['directory'],
                output_path=algo['output'],
                time_column=algo['time_column']
            )

        print("\nProcessing complete.")

        # Create plots after processing
        create_plots()

    elif PROCESSING_MODE == "merge":
        print("Merging mode: Creating master_data.csv from existing cleaned files...")

        # Merge all cleaned CSV files into master_data.csv
        merge_all_csv_files('master_data.csv')

        print("\nMerge complete. Creating plots from master_data.csv...")

        # Create plots using the merged master_data.csv
        create_plots()

    else:  # "use_existing"
        print("Using existing cleaned CSV files for plotting...\n")

        # Create plots using existing cleaned files
        create_plots()


if __name__ == "__main__":
    main()
