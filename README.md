# Yug

Yug is a pretrained time-series foundation model developed by Birla AI Labs
for **zero-shot probabilistic forecasting**. A single checkpoint forecasts a
previously unseen series without task-specific training or per-series tuning:
given a historical context and a forecast horizon, it returns a predictive
distribution summarised by quantile levels.

- Model weights: [`birlaailabs/yug`](https://huggingface.co/birlaailabs/yug)
- Package: `pip install yug`
- Issues and questions: [GitHub Issues](https://github.com/birla-ai-labs/yug/issues)


**Latest version: Yug 0.1.0**

---

## Update — October 2026

First public release of Yug.

**Key highlights:**

- **Zero-shot forecasting.** One pretrained checkpoint forecasts any evenly
  spaced numeric series, across domains and frequencies, without fitting.
- **Probabilistic by default.** Every forecast comes back as quantile levels,
  not a single line, so intervals are available at no extra cost.
- **Fast direct multi-step decoding.** A patch-based decoder emits a whole block
  of future points per forward pass; there is no per-step generation loop.
- **Covariate support.** Optional additional channels alongside the target.
- **Drop-in benchmarking.** A GluonTS predictor for existing evaluation
  harnesses.

---

## Available models

| Model | Parameters | Quantile levels | Context length |
|---|---|---|---|
| [`birlaailabs/yug`](https://huggingface.co/birlaailabs/yug) | 271.8M | 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 | 2048 |

Quantile levels are fixed at training time: a checkpoint cannot emit a level it
was not trained on.

Yug was trained on 2048-point context windows, so by default it reads the
latest 2048 points of each series; shorter series are used whole. This is
also the setting behind its published benchmark scores. Pass
`context_length=...` to change it; feeding more history takes the model
outside its training regime.

---

## Install

From PyPI:

```bash
pip install yug
```

Optional extras:

```bash
pip install 'yug[pandas]'     # used in the example below
pip install 'yug[gluonts]'    # GluonTS predictor for benchmark harnesses
```

Check the machine can run it before downloading weights:

```bash
yug-check
```

Weights are cached locally on first use (`~/.cache/huggingface/hub` by
default, or `HF_HOME` if set).

---

## Code examples

### 1. Forecast one series

```python
import numpy as np
from yug import YugPipeline

pipeline = YugPipeline.from_pretrained(
    "birlaailabs/yug", device_map="cuda"
)

# Synthetic daily signal: trend + weekly/annual seasonality + noise
HORIZON = 48
rng = np.random.default_rng(15)
t = np.arange(1000)
signal = (
    100.0                                    # level
    + 0.05 * t                               # trend
    + 12.0 * np.sin(2 * np.pi * t / 7)      # weekly cycle
    + 25.0 * np.sin(2 * np.pi * t / 365)    # annual cycle
    + rng.normal(0, 3.0, len(t))             # noise
).astype(np.float32)

context, truth = signal[:-HORIZON], signal[-HORIZON:]

forecast = pipeline.predict(
    context,
    prediction_length=HORIZON,
    freq="D",            # a model input, not metadata: pandas offset alias
    seed=0,              # makes the call reproducible
)

print(forecast.median)                # (48,)      point forecast
print(forecast.quantile(0.9))         # (48,)
print(forecast.interval())            # {"lower": (48,), "upper": (48,)}
print(forecast.to_dataframe().head())
```

### 2. Several series at once

Lengths and frequencies may differ.

```python
forecast = pipeline.predict(
    [series_a, series_b, series_c],
    prediction_length=48,
    freq=["D", "D", "H"],
    item_ids=["a", "b", "c"],
)
forecast.values          # (3, n_quantiles, 48)
```

### 3. With covariates

Stack channels into a 2-D array, target first, and pass it as a single series:

```python
multivariate = np.stack([target, covariate_1, covariate_2])
forecast = pipeline.predict([multivariate], prediction_length=48)
```

Covariate forecasts degrade faster with horizon than univariate ones. For long
horizons the univariate path is often the better answer — measure both.

### 4. In a GluonTS benchmark

```python
from yug.gluonts_adapter import YugPredictor

predictor = YugPredictor.from_pretrained(
    "birlaailabs/yug", freq="H", prediction_length=48
)
forecasts = list(predictor.predict(dataset))
```

---

## Examples

- [`notebooks/quickstart.ipynb`](notebooks/quickstart.ipynb) — forecast, plot
  and score a series end to end. Self-contained; generates its own data.

---

## Citation

```bibtex
@software{yug_2026,
  title  = {Yug: a foundation model for zero-shot probabilistic time-series forecasting},
  author = {Aaditya Jain* and Debdeep Sanyal* and Aaryan Nagpal and Dhruv Kumar and Murari Mandal and Saurabh Deshpande},
  year   = {2026},
  organization = {Birla AI Labs},
  url    = {https://github.com/birla-ai-labs/yug},
  license = {Apache-2.0}
}
```

Note: Aaditya Jain and Debdeep Sanyal contributed equally.

## License

Code in this repository is licensed under the
[Apache License 2.0](LICENSE). Model weights on Hugging Face are licensed
separately under a noncommercial license — see the model card for terms.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![CI](https://github.com/birla-ai-labs/yug/actions/workflows/ci.yaml/badge.svg)](https://github.com/birla-ai-labs/yug/actions/workflows/ci.yaml)
