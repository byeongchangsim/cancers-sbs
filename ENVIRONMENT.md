# Computational environments

The result-level check uses the four pinned packages in `requirements.txt`. It was executed with Python 3.8.13, NumPy 1.23.0, pandas 1.5.3, SciPy 1.10.1 and openpyxl 3.1.5.

The recorded revision runtime is in `rev/environment.json` and `rev/runtime_versions.tsv`. These records include PyTorch 2.3.1 with CUDA 12.1, scikit-learn 1.3.2, matplotlib 3.5.3, h5py 3.7.0 and OpenSlide Python 1.3.1. They record the available revision environment and are not presented as an archived lockfile from the original training dates.

The original R setup for deconstructSigs is in `signatures/environment.yml` and `signatures/install_packages.R`. Genome reference packages are specified by the TCGA/CPTAC shell drivers. `records/R_versions.tsv` records the currently installed versions for the retained R environment.

Sigfit 1.2.0 used its historical R 3.4.1 container. The Dockerfile is `signatures/Dockerfile`. Recorded versions are in `records/sigfit_package_versions.tsv`, `records/sigfit_session_info.txt` and `records/sigfit_docker_image.txt`. The Docker build expects the upstream source at commit `27a9ff5ff4f491476815b532564ae1e7955ed001` in `signatures/source/sigfit-v1.2.0`, as specified in the Dockerfile. Obtain that source from the upstream [Sigfit repository](https://github.com/kgori/sigfit) before building.

The saved COSMIC v2 and v3.6 targets were generated with SigProfilerAssignment 1.1.5. The COSMIC v3.5 target driver specifies SigProfilerAssignment 1.1.4. The v2 matrix-generation driver specifies SigProfilerMatrixGenerator 1.3.6. These target-generation environments are separate from the general revision runtime. Installing the older SigProfiler packages listed in that runtime is not a substitute for these recorded target versions.

Frozen RetCCL and CTransPath encoder checkpoints are external inputs. Preserve the recorded architecture and checkpoint hashes. The CTransPath code depends on its compatible timm implementation. See `preprocessing/CTransPath_README.md` and `preprocessing/CTransPath_LICENSE.md`.
