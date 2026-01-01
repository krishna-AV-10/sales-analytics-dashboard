# src/data_processing.py
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


class SalesDataProcessor:
    """
    Robust sales data processor that works with ANY sales dataset:
    - Auto-detects columns
    - Handles any date format
    - Prevents empty-data crashes
    - Safe KPI & insight calculations
    """

    CANONICAL_COLUMNS = {
        "Date": ["date", "order date", "order_date", "transaction date", "time"],
        "Sales": ["sales", "revenue", "amount", "sales amount", "total"],
        "Profit": ["profit", "margin", "net profit"],
        "Quantity": ["quantity", "qty", "units", "count"],
        "Region": ["region", "area", "zone"],
        "Product": ["product", "item", "sku"],
        "Customer": ["customer", "client", "buyer"]
    }

    def __init__(self):
        self.df = None

    # ------------------------------------------------------------------
    # DATA LOADING
    # ------------------------------------------------------------------
    def load_data(self, file_path):
        try:
            if file_path.endswith(".csv"):
                self.df = pd.read_csv(file_path)
            elif file_path.endswith((".xlsx", ".xls")):
                self.df = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format")

            print(f"✅ Loaded {len(self.df)} rows, {len(self.df.columns)} columns")
            return self.df

        except Exception as e:
            print(f"❌ Load error: {e}")
            self.df = None
            return None

    # ------------------------------------------------------------------
    # COLUMN MAPPING
    # ------------------------------------------------------------------
    def _map_columns(self, df):
        rename_map = {}
        for canonical, aliases in self.CANONICAL_COLUMNS.items():
            for col in df.columns:
                if col.lower().strip() in aliases:
                    rename_map[col] = canonical
                    break
        return df.rename(columns=rename_map)

    # ------------------------------------------------------------------
    # DATE PARSING
    # ------------------------------------------------------------------
    def _detect_and_parse_date(self, df):
        for col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
            if parsed.notna().sum() > 0:
                df["Date"] = parsed
                return df
        return df

    # ------------------------------------------------------------------
    # DATA CLEANING
    # ------------------------------------------------------------------
    def clean_data(self):
        if self.df is None or self.df.empty:
            print("❌ No data to clean")
            return None

        df = self.df.copy()

        # Remove duplicates
        df = df.drop_duplicates()

        # Map columns safely
        df = self._map_columns(df)

        # Parse dates safely
        df = self._detect_and_parse_date(df)

        # Numeric safety
        for col in ["Sales", "Profit", "Quantity"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Calculated fields
        if "Sales" in df.columns and "Profit" in df.columns:
            df["Profit_Margin"] = np.where(
                df["Sales"] > 0, (df["Profit"] / df["Sales"] * 100).round(2), 0
            )

        if "Sales" in df.columns and "Quantity" in df.columns:
            df["Unit_Price"] = np.where(
                df["Quantity"] > 0, (df["Sales"] / df["Quantity"]).round(2), 0
            )

        # Time features
        if "Date" in df.columns:
            df = df[df["Date"].notna()]
            if not df.empty:
                df["Year"] = df["Date"].dt.year
                df["Month"] = df["Date"].dt.month_name()
                df["Quarter"] = df["Date"].dt.quarter
                df["Weekday"] = df["Date"].dt.day_name()

        self.df = df
        print(f"✅ Cleaned data shape: {df.shape}")
        return df

    # ------------------------------------------------------------------
    # KPI CALCULATIONS (SAFE)
    # ------------------------------------------------------------------
    def calculate_kpis(self):
        if self.df is None or self.df.empty:
            return {}

        df = self.df
        kpis = {}

        if "Sales" in df.columns:
            kpis["Total_Sales"] = float(df["Sales"].sum())
            kpis["Average_Sales"] = float(df["Sales"].mean())

        if "Profit" in df.columns:
            kpis["Total_Profit"] = float(df["Profit"].sum())
            kpis["Average_Profit"] = float(df["Profit"].mean())

        if "Customer" in df.columns:
            kpis["Unique_Customers"] = int(df["Customer"].nunique())

        if "Region" in df.columns:
            kpis["Regions_Covered"] = int(df["Region"].nunique())

        if "Product" in df.columns:
            kpis["Top_Product"] = (
                df.groupby("Product")["Sales"].sum().idxmax()
                if not df.groupby("Product")["Sales"].sum().empty
                else "N/A"
            )

        if "Date" in df.columns:
            kpis["Date_Range"] = {
                "Start": str(df["Date"].min().date()),
                "End": str(df["Date"].max().date())
            }

        return kpis

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    def get_summary(self):
        if self.df is None:
            return None

        return {
            "Rows": self.df.shape[0],
            "Columns": self.df.shape[1],
            "Column_Names": list(self.df.columns),
            "Missing_Values": self.df.isnull().sum().to_dict()
        }
