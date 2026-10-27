# Rossmann Store Sales API

A live REST API that predicts daily sales for Rossmann drugstores, built from a Random Forest model trained on the Kaggle Rossmann Store Sales data.

**Live demo:** https://rossmann-api-fe31.onrender.com/health

> Hosted on Render's free tier, so the first request after a period of inactivity can take up to a minute while the service wakes up.

## Try it

PowerShell:

```powershell
$body = '{"Store": 1, "Date": "2015-09-17", "Promo": 1, "StateHoliday": "0", "SchoolHoliday": 0}'
Invoke-RestMethod -Uri "https://rossmann-api-fe31.onrender.com/predict" -Method Post -ContentType "application/json" -Body $body
```

curl:

```bash
curl -X POST https://rossmann-api-fe31.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"Store": 1, "Date": "2015-09-17", "Promo": 1, "StateHoliday": "0", "SchoolHoliday": 0}'
```

Example response:

```json
{"predicted_sales": 4698.77}
```

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Service status and model metrics |
| GET | `/schema` | Required and optional input fields, with an example |
| POST | `/predict` | Predict sales for one record, or a list of records for a batch |

**Required fields:** `Store` (1-1115), `Date` (YYYY-MM-DD), `Promo` (0/1), `StateHoliday` (`0`, `a`, `b`, `c`), `SchoolHoliday` (0/1)


**Optional fields:** `Open` (defaults to 1; if 0 the prediction is 0), `Max_Temp`, `Rain_Sum`, `Snow_Sum`

## The model

- **Algorithm:** Random Forest Regressor (51 trees, `min_samples_leaf=15`), predicting `log1p(Sales)` and converting back to sales.
- **Validation RMSPE:** 0.186
- **Training period:** 2013-01-01 to 2015-07-31
- **Features:** store attributes, calendar features, German public holidays, Frankfurt daily weather, competition and Promo2 features, and per-store sales history.

The full analysis, feature engineering and model comparison are in the notebook: [Kahunda/rossmandataset.ipynb](https://github.com/kennethuchegbu22-GOD/Kahunda/blob/main/rossmandataset.ipynb).

The notebook's larger, tuned Random Forest scored 0.1723, but its saved file was 233 MB. This deployed version is a smaller forest trained to fit GitHub's file limit and Render's free-tier memory, at a small cost in accuracy.

## Limitations

- Store sales history comes from 2013-2015, so predictions for dates far beyond that period may drift.
- For dates after 2015-09-17, weather defaults to monthly averages unless you supply `Max_Temp`, `Rain_Sum` and `Snow_Sum`.
- Public holidays are Germany-wide.

## Run locally

```bash
docker build -t rossmann-api .
docker run -p 10000:10000 rossmann-api
```

Then open http://localhost:10000/health.

## Stack

Python, Flask, scikit-learn, pandas, Gunicorn, Docker, Render.
