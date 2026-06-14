Create a Python project that generates synthetic vehicle predictive maintenance datasets for machine learning experiments.

The project should generate the following input files:

1. Vehicle Metadata (vehicle_metadata.csv)

Generate metadata for at least 50 vehicles with the following columns:

vehicle_id

model (Sedan, SUV, Hatchback)

mileage_km

age_years

fuel_type (Petrol, Diesel, Hybrid, EV)

Use realistic distributions for mileage and age.

2. Operational Behavior Dataset (obd_timeseries.csv)

Generate multivariate time-series vehicle diagnostic data sampled at 1 Hz.

Simulate at least 2 hours of driving data per vehicle.

Include these OBD-II/CAN-like signals:

timestamp

vehicle_id

engine_rpm

vehicle_speed_kmh

coolant_temperature_C

throttle_position_pct

engine_load_pct

fuel_level_pct

battery_voltage

mass_airflow_gps

Requirements:

Driving behavior should alternate between:

idle,

city driving,

highway driving,

stop-and-go traffic.

Signals should exhibit realistic correlations.

Higher speed generally corresponds to higher RPM.

Throttle influences RPM and acceleration.

Coolant temperature gradually rises after startup.

Add Gaussian noise to emulate sensor measurements.

3. Fault Injection



Inject synthetic faults into the healthy time-series data.



Randomly assign 1–3 fault events per vehicle.



Implement the following fault types:



Sensor Drift



Gradually increase or decrease a signal over time.

Apply to coolant temperature or battery voltage.



Spikes



Insert short-duration abnormal peaks.

Apply to RPM or MAF signals.



Stuck-at Fault



Freeze a signal at a constant value.

Apply to speed or throttle position.



Increased Noise



Increase measurement variance substantially.

Apply to engine load.



Fault duration should vary between 60 and 600 seconds.



4. Fault Labels (fault_windows.csv)



Create a separate file containing:



vehicle_id

fault_id

fault_type

affected_signal

start_timestamp

end_timestamp



Each injected fault must have a corresponding label entry.



5. DTC Event Logs (dtc_logs.csv)



Generate Diagnostic Trouble Code events associated with injected faults.



Example mappings:



Sensor drift → P0117

RPM spikes → P0300

Speed sensor stuck-at → P0500

Battery faults → P0562



Include columns:



vehicle_id

timestamp

dtc_code

description

fault_id



The DTC timestamp should occur near the end of the corresponding fault window.



6. Labeled Operational Dataset



Add a column named fault_label to the operational dataset.



Possible values:



healthy

sensor_drift

spike

stuck_at

noise



Healthy periods should represent at least 90% of the dataset.



7. Output Requirements



Save the generated datasets as CSV files:



data/

├── vehicle_metadata.csv

├── obd_timeseries.csv

├── fault_windows.csv

├── dtc_logs.csv



Also generate:



README.md



explaining:



dataset structure,

signal definitions,

fault injection methodology,

DTC mappings.

8. Code Requirements

Use Python.

Use pandas, numpy, and standard libraries only.

Organize the solution into reusable functions.

Add type hints and docstrings.

Use a fixed random seed for reproducibility.

Validate that fault windows do not overlap for the same vehicle.

Print summary statistics:

number of vehicles,

total records,

number of faults by type,

number of DTC events.

9. Deliverables



Generate:



generate_dataset.py

Sample CSV outputs in the data/ folder

README.md

Example code demonstrating how to load the datasets into pandas for machine learning experiments.



The generated code should be production-quality, well-commented, and executable without modification using:



python generate_dataset.py



Ensure that all timestamps are internally consistent and that every DTC event can be traced back to a corresponding injected fault.