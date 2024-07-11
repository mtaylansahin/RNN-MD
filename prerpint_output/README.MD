### README for Output Directory

This README file provides an overview of the files and directories included in the extracted output. Each file and its role within the project are described below.

#### File Descriptions

1. **output/Splitting/**
   - This directory contains subdirectories and files related to different data splitting strategies. Each subdirectory corresponds to a specific splitting method used in the analysis.

   - **100-50-50/**: Contains files and results for the 100-50-50 split.
   - **160-20-20/**: Contains files and results for the 160-20-20 split.
   - **50-75-75/**: Contains files and results for the 50-75-75 split.

2. **output/plots/**
   - This directory contains various plot files generated during the analysis.

   - **all_interactions.svg**: An SVG file showing all interactions for 200 time stamps.(hydrophobic, ionics, and hydrogen bond).
   - **bar_all_interactions.png**: A PNG file showing a bar chart of all interactions for 200 time stamps.
   - **bar_chain_A_interactions.png**: A PNG file showing a bar chart of Chain A interactions for 200 time stamps.
   - **heatmap_all.png**: A PNG file showing a heatmap of all interaction for 200 time stamps.
   - **nonpredicted_bar_all_interactions.svg**: An SVG file showing a bar chart of non-predicted interactions for all data.
   - **nonpredicted_bar_chain_A_interactions.svg**: An SVG file showing a bar chart of non-predicted interactions for Chain A.
   - **nonpredicted_ground_truths.svg**: An SVG file showing non-predicted ground truths.
   - **unique_predicted_bar_chain_A_interactions.svg**: An SVG file showing a bar chart of unique predicted interactions for Chain A.
   - **unique_predicted_bar_chain_C_interactions.svg**: An SVG file showing a bar chart of unique predicted interactions for Chain C.

3. **output/data_preperation_for_network_analysis_of_1JPS.ipynb**
   - A Jupyter Notebook file containing the code and documentation for preparing data for network analysis of the 1JPS dataset.

4. **output/labels.txt**
   - A text file containing labels used in the analysis. This file is likely used for mapping or referencing data labels in the project.

7. **output/tables**
   -  **unique_values_not_in_ground_truth.xlsx**: This file contains information about false positive predictions. 
   -  **unpredictable_ground_truth.xlsx**: This file contains information about false negative predictions.

### Detailed Explanation of Files under "Splitting" Directories

#### 1. **Each splitting directory (100-50-50, 160-20-20, 50-75-75) contains following main subdirectories:**

These directories collectively provide comprehensive data and results for various data split strategies used in the analysis.
Each splitting directory (100-50-50, 160-20-20, 50-75-75) contains four main subdirectories:

- **DFs_split-xx-xx-xx/**: Directory containing data frames for the xx-xx-xx split.
  - **train.txt**: Training data.
  - **valid.txt**: Validation data.
  - **test.txt**: Testing data.
  - **stat.txt**: Statistical summary of the data.

- **output_1JPS_split_xx-xx-xx/**: Directory containing output files for the xx-xx-xx split.
  - **result.py**: Python script for analyzing results.
  - **bubble_output_renet_1JPS_x.txt.png**: Bubble plot for x parameter
  - **scatter_output_renet_1JPS_x.txt.png**: Scatter plot for x parameter.
  - **scores_output_renet_1JPS_x.txt**: Scores for x parameter.

- **valid_1JPS_split_xx-xx-xx/**: Directory containing validation files for xx-xx-xx split.
  - **metrics_valid**: Performance metrixs for Validation.
  - **output_renet_1JPS_x.txt**: Output for x parameter.

- **graphs_and_tables_for_optimized_hyperparameter_xx-xx-xx/graphs/**

   This directory contains various graphical representations of the analysis results for the xx-xx-xx data split.

   - **Missings_xx-xx-xx.png**: Plot showing missing values in the dataset.
   - **bar_all_xx-xx-xx.png**: PNG file with a bar chart of all interactions.
   - **bar_all_xx-xx-xx.svg**: SVG file with a bar chart of all interactions.
   - **bar_in_ground_truth_xx-xx-xx.png**: PNG file with a bar chart of interactions in the ground truth.
   - **bar_in_ground_truth_xx-xx-xx.svg**: SVG file with a bar chart of interactions in the ground truth.
   - **bar_in_predicted_xx-xx-xx.png**: PNG file with a bar chart of predicted interactions.
   - **bar_in_predicted_xx-xx-xx.svg**: SVG file with a bar chart of predicted interactions.
   - **heatmap_xx_xx_xx.png**: Heatmap of the interactions for the xx-xx-xx split.
   - **time_xx-xx-xx.png**: PNG file showing time analysis for the xx-xx-xx split.
   - **time_xx-xx-xx.svg**: SVG file showing time analysis for the xx-xx-xx split.

- **graphs_and_tables_for_optimized_hyperparameter_xx-xx-xx/tables/**

   This directory contains tables and spreadsheets with detailed data for the xx-xx-xx data split.

   - **xx-xx-xx_ground_truth.xlsx**: Excel file with the ground truth interactions for the xx-xx-xx split.
   - **xx-xx-xx_ground_truth.txt**: Text file with the ground truth interactions for the xx-xx-xx split.
   - **xx-xx-xx_prediction.xlsx**: Excel file with the predicted interactions for the xx-xx-xx split.
   - **xx-xx-xx_prediction.txt**: Text file with the predicted interactions for the xx-xx-xx split.
   - **nonpredicted_interactions_xx-xx-xx.xlsx**: Excel file with interactions that were not predicted in the xx-xx-xx split.
   - **performance_xx_xx_xx_test_set.xlsx**: Excel file with performance metrics for the test set of the xx-xx-xx split.
   - **performance_xx_xx_xx_validation_set.xlsx**: Excel file with performance metrics for the validation set of the xx-xx-xx split.
   - **time_xx-xx-xx.png**: PNG file showing time analysis for the xx-xx-xx split.
   - **unique_predicted_interactions_xx-xx-xx.xlsx**: Excel file with unique predicted interactions for the xx-xx-xx split.






