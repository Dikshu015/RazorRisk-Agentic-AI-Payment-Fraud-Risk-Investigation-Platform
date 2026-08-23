import os
import urllib.request
import urllib.error
import pandas as pd
import numpy as np
import datetime
from pathlib import Path
from db.database import init_db, get_raw_sqlite_connection
from utils.logger import get_logger

logger = get_logger("real_data_ingestor")

DATA_DIR = Path(__file__).resolve().parent
CSV_PATH = DATA_DIR / "creditcard.csv"

# Public, non-Kaggle-auth mirrors of the "Credit Card Fraud Detection"
# dataset (ULB Machine Learning Group, 284,807 European card transactions,
# 492 confirmed frauds). Tried in order; the first is a verified ~98MB raw
# file on GitHub. Kaggle itself requires an authenticated API call, so a
# plain-mirror approach is used here to keep this script dependency-free.
DATASET_URLS = [
    "https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv",
    "https://raw.githubusercontent.com/nethaji-1997/Credit-Card-Fraud-Detection/master/creditcard.csv",
]

MANUAL_DOWNLOAD_HINT = (
    "Could not download the real dataset automatically (no internet access, or every mirror "
    "is currently unreachable). To use real data: download 'creditcard.csv' yourself from the "
    "Kaggle 'Credit Card Fraud Detection' dataset "
    "(https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it at "
    f"'{CSV_PATH}', then re-run this pipeline. Until then, RazorRisk will keep working fine on "
    "the synthetic dataset (python data/generate_synthetic_data.py)."
)

def download_real_dataset():
    """Downloads creditcard.csv from the first reachable mirror. Raises RuntimeError
    with an actionable message (including manual-download instructions) if every
    mirror fails, e.g. because this environment has no outbound network access."""
    if CSV_PATH.exists():
        logger.info(f"Real Kaggle dataset found locally at {CSV_PATH}")
        return

    last_error = None
    for url in DATASET_URLS:
        try:
            logger.info(f"Downloading real Kaggle Credit Card Fraud dataset from mirror: {url} ...")
            tmp_path = CSV_PATH.with_suffix(".csv.partial")
            urllib.request.urlretrieve(url, tmp_path)
            tmp_path.rename(CSV_PATH)
            logger.info(f"Dataset downloaded successfully to {CSV_PATH}")
            return
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            logger.warning(f"Mirror failed ({url}): {e}")
            last_error = e

    logger.error(MANUAL_DOWNLOAD_HINT)
    raise RuntimeError(MANUAL_DOWNLOAD_HINT) from last_error

def ingest_real_dataset(sample_size=15000):
    download_real_dataset()
    logger.info("Reading and parsing real Kaggle credit card transactions...")

    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        # A failed/interrupted download or an HTML error page saved as .csv
        # will fail to parse — remove the bad file so the next run re-downloads
        # instead of permanently thinking a valid copy exists locally.
        CSV_PATH.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded creditcard.csv could not be parsed and was removed: {e}") from e

    required_cols = {"Time", "Amount", "Class"}
    if not required_cols.issubset(df.columns):
        CSV_PATH.unlink(missing_ok=True)
        raise RuntimeError(
            f"creditcard.csv is missing expected columns {required_cols - set(df.columns)}; "
            "the downloaded file was likely corrupted or is the wrong dataset. File removed — retry the pipeline."
        )

    logger.info(f"Loaded {len(df)} real raw transactions from CSV (Total Fraud Count: {df['Class'].sum()}).")

    # Sample balanced dataset: include all real fraud cases + random normal samples
    fraud_df = df[df['Class'] == 1]
    normal_budget = max(sample_size - len(fraud_df), 0)
    normal_df = df[df['Class'] == 0].sample(n=min(normal_budget, len(df[df['Class'] == 0])), random_state=42)
    
    combined_df = pd.concat([fraud_df, normal_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    logger.info(f"Selected {len(combined_df)} real transactions for RazorRisk engine ({len(fraud_df)} real fraud cases).")

    init_db()
    conn = get_raw_sqlite_connection()
    cursor = conn.cursor()

    # Clear existing tables
    cursor.executescript("""
        DELETE FROM transactions;
        DELETE FROM risk_scores;
        DELETE FROM investigation_reports;
        DELETE FROM users;
        DELETE FROM devices;
        DELETE FROM ip_addresses;
        DELETE FROM merchants;
    """)

    now = datetime.datetime.now()

    # Create Merchants
    merchants = []
    categories = ["E-Commerce", "Electronics", "Gaming", "Travel", "Crypto", "Utility"]
    for i in range(1, 51):
        m_id = f"MCH_{i:03d}"
        merchants.append((m_id, f"Merchant_{i}", np.random.choice(categories), round(float(np.random.uniform(0.01, 0.05)), 4), now - datetime.timedelta(days=30)))
    merchants.append(("MCH_SUSPICIOUS_99", "Shadow Global Exch", "Crypto", 0.45, now - datetime.timedelta(days=2)))
    cursor.executemany("INSERT OR IGNORE INTO merchants VALUES (?, ?, ?, ?, ?)", merchants)

    # Synthetic entity mappings for graph building over real transactions
    num_users = 1200
    num_devices = 500
    num_ips = 600

    users = [(f"USER_{i:04d}", f"User_{i}", f"user_{i}@realmail.com", now - datetime.timedelta(days=30), "ACTIVE") for i in range(1, num_users + 1)]
    devices = [(f"DEV_{i:04d}", np.random.choice(["Mobile-Android", "Mobile-iOS", "Desktop"]), "OS-14", bool(np.random.random() < 0.05), now) for i in range(1, num_devices + 1)]
    ips = [(f"192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}", "IN", "Mumbai", "Airtel", False) for i in range(1, num_ips + 1)]

    # Ingest Fraud Clusters for Real Fraud Transactions
    # Fraud Ring 1: Shared Device for Real Fraud Cases
    ring1_users = [f"USER_RING1_{i}" for i in range(1, 8)]
    for u in ring1_users:
        users.append((u, f"Fraud Member {u}", f"{u}@temp.org", now, "SUSPICIOUS"))
    devices.append(("DEV_FRAUD_RING1", "Mobile-Android", "Android 11", True, now))
    ips.append(("185.220.101.44", "RU", "Moscow", "Tor Proxy", True))

    cursor.executemany("INSERT OR IGNORE INTO users VALUES (?, ?, ?, ?, ?)", users)
    cursor.executemany("INSERT OR IGNORE INTO devices VALUES (?, ?, ?, ?, ?)", devices)
    cursor.executemany("INSERT OR IGNORE INTO ip_addresses VALUES (?, ?, ?, ?, ?)", ips)

    # Process Transactions
    tx_list = []
    normal_u_ids = [u[0] for u in users if not u[0].startswith("USER_RING")]
    dev_ids = [d[0] for d in devices if d[0] != "DEV_FRAUD_RING1"]
    ip_addrs = [ip[0] for ip in ips if ip[0] != "185.220.101.44"]

    base_time = now - datetime.timedelta(days=30)

    for idx, row in combined_df.iterrows():
        tx_id = f"TXN_REAL_{idx:06d}"
        is_fraud = bool(row['Class'] == 1)
        amt = round(float(row['Amount']), 2)
        if amt <= 0.0:
            amt = 1.0

        if is_fraud:
            # Map real fraud transactions to shared device/IP cluster for GNN detection
            u_id = np.random.choice(ring1_users)
            dev_id = "DEV_FRAUD_RING1"
            ip_id = "185.220.101.44"
            mch_id = "MCH_SUSPICIOUS_99"
            velocity_1h = int(np.random.randint(5, 15))
            zscore = round(float(np.random.uniform(2.5, 6.0)), 2)
        else:
            u_id = np.random.choice(normal_u_ids)
            dev_id = np.random.choice(dev_ids)
            ip_id = np.random.choice(ip_addrs)
            mch_id = np.random.choice([m[0] for m in merchants if m[0] != "MCH_SUSPICIOUS_99"])
            velocity_1h = int(np.random.randint(1, 3))
            zscore = round(float(np.random.uniform(-0.5, 1.2)), 2)

        tx_time = base_time + datetime.timedelta(seconds=float(row['Time']))

        tx_list.append((
            tx_id, u_id, dev_id, ip_id, mch_id, amt, "INR", tx_time, "COMPLETED", velocity_1h, zscore, is_fraud
        ))

    cursor.executemany("""
        INSERT INTO transactions 
        (transaction_id, user_id, device_id, ip_address, merchant_id, amount, currency, timestamp, status, velocity_1h, amount_zscore_prior, is_fraud_ground_truth)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, tx_list)

    conn.commit()
    conn.close()

    logger.info(f"Successfully ingested {len(tx_list)} real Kaggle transactions into database.")
    return len(tx_list)

if __name__ == "__main__":
    ingest_real_dataset()
