# nyc-airbnb-mini-project-1

Interactive Streamlit dashboard exploring the [NYC Airbnb Open Data (2019)](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data).

- **Live dashboard:** ✍️ _Streamlit Cloud link_
- **Central question:** ✍️ …

## Faithfulness & Granularity

- **What each row represents:** ✍️ …
- **Scope of the data:** ✍️ …
- **Strange values, inconsistencies, duplicates:** ✍️ …

## Process

✍️ Cleaning choices, features, model, and the key finding.

## Running locally

Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync                          # create .venv and install everything
uv run streamlit run app.py      # launch the dashboard
```

The analysis notebook is `main.ipynb`; select the `.venv` kernel.

### Adding dependencies

`requirements.txt` is generated from `uv.lock` (Streamlit Cloud installs from it). After any `uv add` / `uv remove`, regenerate it:

```bash
uv export --no-hashes --no-dev --no-emit-project --format requirements-txt -o requirements.txt
```

Notebook-only tools (`ipykernel`, `nbconvert`) live in the `dev` group so they stay out of the deployed app.

## Deployment

Deployed on Streamlit Community Cloud: main file `app.py`, Python 3.12 (Advanced settings). The app downloads the dataset with `kagglehub` on first load; to skip that, commit the CSV to `data/AB_NYC_2019.csv` and it will be used instead.
