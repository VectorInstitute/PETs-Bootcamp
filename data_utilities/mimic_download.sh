#!/bin/bash
cd ~
source /ssd003/projects/aieng/public/mimic_preprocessing/bin/activate
#Download dataset, extract relevant files, delete excess and unzip files mkdir mimic3
#Replace USERNAME with PhysioNet username
wget -r -N -c -np --user USERNAME --ask-password https://physionet.org/files/mimiciii/1.4/
mv physionet.org/files/mimiciii/1.4 mimic3
rm -r physionet.org
gunzip mimic3/1.4/*.gz

#Clone repository with utilities to load and process
git clone https://github.com/jewelltaylor/mimic3-benchmarks.git
#Generates one directory per SUBJECT_ID and writes ICU stay information
(cd mimic3-benchmarks && python -m mimic3benchmark.scripts.extract_subjects ../mimic3/1.4 data/root/)
#Breaks up per-subject data into separate episodes (pertaining to ICU stays)
(cd mimic3-benchmarks && python -m mimic3benchmark.scripts.extract_episodes_from_subjects data/root/)
# splits the whole dataset into training and testing sets.
(cd mimic3-benchmarks && python -m mimic3benchmark.scripts.split_train_and_test data/root/)
# Generate In Hospital Mortality Prediction
(cd mimic3-benchmarks && python -m mimic3benchmark.scripts.create_in_hospital_mortality data/root/ data/in-hospital-mortality/)
# Split train into train and valdation
(cd mimic3-benchmarks && python -m mimic3models.split_train_val data/in-hospital-mortality)
# Save Date to numpy
(cd mimic3-benchmarks && python -um mimic3models.in_hospital_mortality.save_mimic_data)
