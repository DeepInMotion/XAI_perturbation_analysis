# XAI Perturbation Analysis Tool

This repository contains scripts for performing XAI-informed perturbation analysis on our CP prediction model. It analyzes how modifying the velocity or angle of different body keypoints affects classification outcomes.

## Overview

The XAI Perturbation Analysis Tool systematically modifies kinematic data (joint positions over time) to understand how our model makes decisions. It focuses on identifying which body keypoints most strongly influence model predictions when their speed/angle is modified.

## Features

- Systematically perturb joint motion speeds and angles to analyze model response
- Separate analysis for important (top-k) and non-important (non-top-k) joints
- Automatic scaling factor calculation based on velocity/angular distributions
- Excel output of results

## Installation

Place the perturbation scripts in the CP prediction model folder:
   ```
   cp perturb_angle.py perturb_velocity.py /path/to/cp_prediction_model/
   ```

## Data Structure

The tool expects the following data organization within our CP prediction model directory:

```
./data/
  ├── tracked_*.csv                         # CSV files containing joint tracking data (pre-selected windows from test data)
  ├── video_fps_dict.json                   # Dictionary mapping video filenames to frame rates
  ├── cams_pvb_test.pkl                     # Pickled test data with CAM values and PVB input to model
  ├── percentiles_cam_lowrisk_trainval.pkl  # Velocity percentiles data
  ├── min_max_angles_trainval.pkl           # Angular percentiles reference data
  └── filename_window_lowrisk.pkl           # Predefined windows for angular perturbation analysis
```

The CSV tracking files should contain joint position data for 19 body parts with the following format:
- Each row represents a video frame
- Each body part has an x and y coordinate column
- Total of 38 columns (19 joints × 2 coordinates)

## Usage

Once the python files are in the CP prediction model directory, run:

```
# For velocity-based perturbation analysis
python perturb_velocity.py

# For angle-based perturbation analysis
python perturb_angle.py
```

Each script will:
1. Load the tracking data and respective statistical references
2. Process the predefined windows for analysis
3. Perform perturbation analysis on both top-k and non-top-k joints
4. Generate Excel files with the results and highlighted risk predictions

## Output

The modules generate several output files:

- `cam_topk_lowrisk_test.pkl`: Perturbation results for top-k segments
- `cam_nontopk_lowrisk_test.pkl`: Perturbation results for non-top-k segments
- `cam_topk_lowrisk_test.xlsx`: Formatted Excel file with top-k results
- `cam_nontopk_lowrisk_test.xlsx`: Formatted Excel file with non-top-k results

## How It Works

1. **Window Selection**: The tools use predefined windows from different video files.

2. **Scaling Factor Calculation**: 
   - **Velocity Perturbation**: Calculates speed scaling factors based on velocity distributions.
   - **Angle Perturbation**: Calculates angle modification factors based on angular change distributions.

3. **Perturbation**: Both tools perform two types of perturbations:
   - Reducing angle/velocity (using factors 0.2, 0.25, 1/3, 0.5, 1)
   - Increasing angle/velocity (using factors 1, 2, 3, 4, 5)

4. **Analysis**: For each perturbation, the modified data is fed to the classification model and changes in prediction are recorded.

5. **Results**: Results are saved to Excel files with formatting to highlight risk thresholds.

## Citation

If you use this tool in your research, please cite:

```
@article{pellano2025towards,
  title={Towards Biomarker Discovery for Early Cerebral Palsy Detection: Evaluating Explanations Through Kinematic Perturbations},
  author={Pellano, Kimji N and Str{\"u}mke, Inga and Groos, Daniel and Adde, Lars and Haugen, P{\aa}l and Ihlen, Espen Alexander F},
  journal={arXiv preprint arXiv:2503.16452},
  year={2025}
}
```
