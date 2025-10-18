from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine
import pandas as pd
import os
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
import numpy as np

# Load environment variables
load_dotenv()

app = FastAPI()

# Allow frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# PostgreSQL connection
DATABASE_URL = os.getenv("DATABASE_URL")  # e.g. postgresql://postgres:admin@localhost:5432/budget_planner
engine = create_engine(DATABASE_URL)


@app.get("/")
def root():
    return {"message": "AI Budget Planner API is running!"}


@app.post("/upload-csv")
def upload_csv():
    try:
        df = pd.read_csv("../data/financial_dataset.csv")
        df.to_sql("cost_items", engine, if_exists="replace", index=False)
        return {"status": "CSV uploaded successfully!"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/data")
def get_data():
    try:
        df = pd.read_sql("SELECT * FROM cost_items;", engine)
        # Normalize columns: look for common spend column names
        spend_col = None
        for col in df.columns:
            if "amount" in col.lower() or "spend" in col.lower() or "cost" in col.lower():
                spend_col = col
                break

        if not spend_col:
            return {"error": "No numeric spend column found in dataset."}

        # Prepare summarized monthly data
        if "month" in df.columns:
            monthly_data = df.groupby("month")[spend_col].sum().reset_index()
        elif "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df["month"] = df["date"].dt.strftime("%b")
            monthly_data = df.groupby("month")[spend_col].sum().reset_index()
        else:
            # fallback to sample months
            months = ["Jan", "Feb", "Mar", "Apr", "May"]
            monthly_data = pd.DataFrame({"month": months[:len(df)], spend_col: df[spend_col].head(len(months)).values})

        return monthly_data.to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}


@app.get("/forecast")
def forecast_next_month():
    try:
        df = pd.read_sql("SELECT * FROM cost_items;", engine)

        # Detect numeric spend column
        spend_col = None
        for col in df.columns:
            if "amount" in col.lower() or "spend" in col.lower() or "cost" in col.lower():
                spend_col = col
                break

        if not spend_col:
            return {"error": "No spend column found"}

        # Prepare numeric sequence for training
        y = df[spend_col].astype(float).values[:5]
        X = np.arange(len(y)).reshape(-1, 1)

        model = LinearRegression()
        model.fit(X, y)

        next_month = model.predict(np.array([[len(y)]]))[0]
        return {
            "forecast_next_month": round(float(next_month), 2),
            "trend": "increasing" if model.coef_[0] > 0 else "decreasing",
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/anomalies")
def detect_anomalies():
    try:
        df = pd.read_sql("SELECT * FROM cost_items;", engine)
        spend_col = None
        for col in df.columns:
            if "amount" in col.lower() or "spend" in col.lower() or "cost" in col.lower():
                spend_col = col
                break

        if not spend_col:
            return {"error": "No spend column found"}


        mean = df[spend_col].mean()
        std = df[spend_col].std()
        threshold = mean + 2 * std

        anomalies = df[df[spend_col] > threshold]
        return {"anomalies": anomalies.to_dict(orient="records")}
    except Exception as e:
        return {"error": str(e)}
