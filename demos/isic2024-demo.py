import pandas as pd
import gradio as gr
import numpy as np
import os

# Define paths for both train and test metadata
train_csv_path = "/mnt/nvme-fast0/datasets/isic-2024-challenge/train-metadata.csv"
test_csv_path = "/mnt/nvme-fast0/datasets/isic-2024-challenge/test-metadata.csv"

# Load both datasets
train_metadata = pd.read_csv(train_csv_path)
test_metadata = pd.read_csv(test_csv_path)


def view_dataframe(dataset, num_rows):
    if dataset == "Train":
        return train_metadata.head(num_rows)
    else:
        return test_metadata.head(num_rows)


with gr.Blocks(theme="huggingface") as iface:
    gr.Markdown("# Pandas DataFrame Viewer")
    gr.Markdown(
        "Use the dropdown to select the dataset and the slider to choose the number of rows to display."
    )

    with gr.Row():
        dataset_choice = gr.Dropdown(["Train", "Test"], label="Select Dataset", value="Train")
        num_rows = gr.Slider(
            1,
            max(len(train_metadata), len(test_metadata)),
            step=1,
            value=10,
            label="Number of rows to display",
        )

    output_table = gr.DataFrame()

    def update_table(dataset, num_rows):
        return view_dataframe(dataset, num_rows)

    dataset_choice.change(update_table, inputs=[dataset_choice, num_rows], outputs=output_table)
    num_rows.change(update_table, inputs=[dataset_choice, num_rows], outputs=output_table)

if __name__ == "__main__":
    iface.launch(share=True, debug=True)
