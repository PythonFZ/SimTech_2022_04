[![DVC](https://img.shields.io/badge/-tracked-white.svg?logo=data-version-control&link=https://dvc.org/?utm_campaign=badge)](https://studio.iterative.ai/user/PythonFZ/views/SimTech_2022_04-gwu9ba191d)
 [![ZnTrack](https://img.shields.io/badge/Powered%20by-ZnTrack-%23007CB0)](https://zntrack.readthedocs.io/en/latest/)
 [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/PythonFZ/SimTech_2022_04/HEAD)
[![DagsHub](https://bit.ly/33Q1Dv9)](https://dagshub.com/PythonFZ/SimTech_2022_04)
# CML Worflow for Training a CNN Model using DVC/ZnTrack

This repository contains the CML configuration as well as the Node configuration for the following Workflow.
It uses public GitHub Runners and a minio S3 storage.

```mermaid
graph TD

A[Download Kaggle Dataset] --> B[Preprocessing Training Data]
A --> C[Preprocessing Test Data]

B -->D[Train ML Model]

D --> E[Evaluate ML Model]
C --> E

```

Learn more about https://www.simtech.uni-stuttgart.de/
