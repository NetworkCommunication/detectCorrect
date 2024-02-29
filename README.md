# detectCorrect

## Project Introduction

This project introduces a collaborative scheme involving vehicles, Road Side Units (RSUs), and Data Center (DC) to jointly enhance the accuracy of vehicle-transmitted BSMs. Our project involves analyzing statistical features of vehicle driving information to detect error BSMs. These detected errors are subsequently corrected by leveraging historical data from the vehicle and its relative relationship with surrounding vehicles. In addition, we propose a time optimization method to reduce the average processing time of each data by RSUs. The extensive experimental results demonstrate that the proposed scheme can accurately detect error BSMs and effectively correct error BSMs. The entire scheme also meets the requisite computational latency requirements

## Environmental Dependence

The code requires python3 (>=3.8) with the development headers. The code also need system packages as bellow:

numpy == 1.24.3

matplotlib == 3.2.2

pandas == 2.0.3

python == 3.8.18

tensorboard == 2.13.0

gym == 0.22.0

tensorflow == 2.13.0

pyqt == 5.15.10

If users encounter environmental problems and reference package version problems that prevent the program from running, please refer to the above installation package and corresponding version.

## Project Structure Introduction and How to Run

The first step in running the algorithm involves processing the original dataset, for which the code can be found in "process_data.py". After the data has been processed, you can run either "main.py" or "my_main.py". The former demonstrates the results without adopting a time optimization scheme, while the latter shows the effects of using a time optimization method. "Class_DataCenter0413.py, Class_rsu_0413.py, and Class_vehicle.py" are used to simulate the data center, RSU, and vehicles, respectively. The final analysis of the experiment results can be referred to in "result_analyze.py and my_result_analyze.py".

## Statement

If you want to know more, please refer to our paper "A Collaborative Error Detection and Correction Scheme for Safety Message in V2X".