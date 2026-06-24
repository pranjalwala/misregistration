\# Misregistration Benchmark



A benchmark package for measuring and analyzing CMYK channel misregistration.



\## Installation



```bash

pip install -e .

```



\## Run Tests



```bash

python -m pytest -v

```



\## Quick Start



Smoke test:



```bash

python scripts/smoke\_test.py --demo

```



Run on a dataset:



```bash

python scripts/run\_dataset\_benchmark.py data --dpi 300

```



\## Repository Structure



```text

misregistration/

scripts/

tests/

configs/

examples/



README.md

pyproject.toml

```



\## Outputs



Generated outputs include:



\* Per-image measurements

\* Per-method summaries

\* Misregistration statistics

\* ΔE maps

\* Vector field visualizations

\* Benchmark tables

\* Pareto analysis results



Outputs are written to:



```text

results/

```



