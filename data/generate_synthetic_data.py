import random
import datetime
import uuid
import pandas as pd
import numpy as np
from db.database import init_db, get_raw_sqlite_connection
from utils.logger import get_logger

logger = get_logger("data_generator")

def generate_dataset(num_users=1500, num_transactions=12000, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    logger.info("Generating synthetic payments dataset with fraud rings...")

    init_db()
    conn = get_raw_sqlite_connection()
    cursor = conn.cursor()

    # 1. Clear existing data
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

    # 2. Generate Merchants
    merchants = []
    categories = ["E-Commerce", "Electronics", "Gaming", "Travel", "Crypto", "Utility", "Food & Dining"]
    for i in range(1, 51):
        m_id = f"MCH_{i:03d}"
        name = f"Merchant_{i} ({random.choice(categories)})"
        cat = random.choice(categories)
        fraud_rate = 0.15 if cat == "Crypto" or i == 42 else round(random.uniform(0.005, 0.03), 4)
        merchants.append((m_id, name, cat, fraud_rate, now - datetime.timedelta(days=random.randint(30, 365))))
    
    # Injected Suspicious Merchant
    merchants.append(("MCH_SUSPICIOUS_99", "Shadow Global Exch", "Crypto", 0.45, now - datetime.timedelta(days=2)))
    
    cursor.executemany("INSERT OR IGNORE INTO merchants VALUES (?, ?, ?, ?, ?)", merchants)
    logger.info(f"Inserted {len(merchants)} merchants.")

    # 3. Generate Normal Users — each with THEIR OWN dedicated device + IP.
    # (Earlier versions drew a random device and random IP per transaction
    # from a shared pool of 600 devices / 800 IPs across 1500 users — with
    # 12k transactions that meant every device/IP coincidentally ended up
    # "shared" by several users just by chance, producing a near-complete
    # 190k-edge graph instead of a sparse one. A real user has one phone;
    # modeling that 1:1 is what makes "shared device" a meaningful signal
    # at all instead of noise.)
    users = []
    devices = []
    ips = []
    user_device = {}  # user_id -> device_id (their own)
    user_ip = {}       # user_id -> ip_address (their own; may gain a 2nd via co-location)

    for i in range(1, num_users + 1):
        u_id = f"USER_{i:04d}"
        name = f"User_{i}"
        email = f"user_{i}@example.com"
        users.append((u_id, name, email, now - datetime.timedelta(days=random.randint(10, 500)), "ACTIVE"))

        d_id = f"DEV_{i:04d}"
        d_type = random.choice(["Mobile-Android", "Mobile-iOS", "Desktop-Windows", "Desktop-Mac"])
        os_sys = random.choice(["Android 14", "iOS 17", "Windows 11", "macOS Sonoma"])
        is_vpn = random.random() < 0.05
        devices.append((d_id, d_type, os_sys, is_vpn, now - datetime.timedelta(days=random.randint(10, 300))))
        user_device[u_id] = d_id

        ip_addr = f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}"
        country = "IN" if random.random() < 0.9 else random.choice(["US", "SG", "GB", "RU", "CN"])
        city = random.choice(["Mumbai", "Bangalore", "Delhi", "Hyderabad", "Chennai", "Pune"])
        isp = random.choice(["Airtel", "Jio", "Vodafone", "ACT Fiber"])
        is_proxy = country in ["RU", "CN"]
        ips.append((ip_addr, country, city, isp, is_proxy))
        user_ip[u_id] = ip_addr

    normal_u_ids = list(user_device.keys())

    # Benign co-locations (noise): pairs of normal users who happen to share
    # ONE IP (roommates, office wifi, a cafe) with no elevated velocity and
    # different devices — no fraud behavior at all. Without this, "any
    # shared IP" would be a perfect fraud signal by construction, which is
    # too easy: it gives the GNN/tabular model nothing real to beat. With
    # it, a naive shared-IP-only rule produces false positives, same as a
    # real payments graph.
    num_colocations = max(1, num_users // 25)
    colocation_pairs = 0
    used_in_colocation = set()
    attempts = 0
    user_secondary_ip = {}  # user_id -> an occasional second IP (a coworker's/roommate's)
    while colocation_pairs < num_colocations and attempts < num_colocations * 20:
        attempts += 1
        u1, u2 = random.sample(normal_u_ids, 2)
        if u1 in used_in_colocation or u2 in used_in_colocation:
            continue
        user_secondary_ip[u2] = user_ip[u1]
        used_in_colocation.add(u1)
        used_in_colocation.add(u2)
        colocation_pairs += 1
    logger.info(f"Prepared {colocation_pairs} benign IP co-locations (noise, not fraud).")

    # 4. Create Injected Fraud Rings
    # Fraud Ring 1: Device Sharing Ring (7 accounts, 1 device, 1 IP)
    ring1_users = [f"USER_RING1_{i}" for i in range(1, 8)]
    for u in ring1_users:
        users.append((u, f"Fraud Ring1 Member {u}", f"{u}@tempmail.com", now - datetime.timedelta(days=1), "SUSPICIOUS"))
    devices.append(("DEV_FRAUD_RING1", "Mobile-Android", "Android 11", True, now - datetime.timedelta(hours=5)))
    ips.append(("185.220.101.44", "RU", "Moscow", "Tor Proxy Node", True))

    # Fraud Ring 2: IP Velocity / Proxy Cluster (8 accounts, 1 shared IP)
    ring2_users = [f"USER_RING2_{i}" for i in range(1, 9)]
    for u in ring2_users:
        users.append((u, f"Fraud Ring2 Member {u}", f"{u}@anonymous.io", now - datetime.timedelta(days=2), "SUSPICIOUS"))
        devices.append((f"DEV_RING2_{u}", "Mobile-iOS", "iOS 16", True, now - datetime.timedelta(days=2)))
    ips.append(("198.51.100.99", "CN", "Beijing", "High Risk VPN", True))

    # Fraud Ring 3: High Velocity Carding Attack (1 user, 1 device, rapid txns)
    ring3_user = "USER_CARDER_X"
    users.append((ring3_user, "Attacker Carder X", "carderx@darknet.net", now - datetime.timedelta(days=1), "SUSPICIOUS"))
    devices.append(("DEV_CARDER_X", "Desktop-Windows", "Windows 10", True, now - datetime.timedelta(hours=2)))
    ips.append(("203.0.113.50", "IN", "Mumbai", "Public Wi-Fi", False))

    # Fraud Ring 4: Suspicious Merchant Collusion (5 users -> MCH_SUSPICIOUS_99)
    ring4_users = [f"USER_COLLUSION_{i}" for i in range(1, 6)]
    ring4_ip = {}
    for u in ring4_users:
        users.append((u, f"Collusion User {u}", f"{u}@mulemail.org", now - datetime.timedelta(days=3), "SUSPICIOUS"))
        devices.append((f"DEV_RING4_{u}", "Mobile-Android", "Android 13", False, now - datetime.timedelta(days=3)))
        ip_addr = f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}"
        ips.append((ip_addr, "IN", random.choice(["Mumbai", "Delhi", "Pune"]), "Jio", False))
        ring4_ip[u] = ip_addr

    cursor.executemany("INSERT OR IGNORE INTO users VALUES (?, ?, ?, ?, ?)", users)
    cursor.executemany("INSERT OR IGNORE INTO devices VALUES (?, ?, ?, ?, ?)", devices)
    cursor.executemany("INSERT OR IGNORE INTO ip_addresses VALUES (?, ?, ?, ?, ?)", ips)
    logger.info(f"Inserted {len(users)} users, {len(devices)} devices, {len(ips)} IPs.")

    # 5. Generate Transactions
    tx_list = []
    mch_ids = [m[0] for m in merchants if m[0] != "MCH_SUSPICIOUS_99"]

    # Generate normal transactions spread over past 30 days — each always
    # uses ITS OWN user's device + IP (occasionally the co-located secondary
    # IP), not an independent random draw, since a person doesn't switch
    # phones between purchases.
    base_time = now - datetime.timedelta(days=30)
    for i in range(1, num_transactions + 1):
        tx_id = f"TXN_NORM_{i:06d}"
        u = random.choice(normal_u_ids)
        d = user_device[u]
        ip = user_secondary_ip[u] if (u in user_secondary_ip and random.random() < 0.3) else user_ip[u]
        m = random.choice(mch_ids)
        amt = round(float(np.random.exponential(scale=1200) + 150), 2)
        if amt > 25000:
            amt = round(random.uniform(500, 4500), 2)
        
        tx_time = base_time + datetime.timedelta(minutes=random.randint(1, 30*24*60))
        tx_list.append((
            tx_id, u, d, ip, m, amt, "INR", tx_time, "COMPLETED", 1, round(random.uniform(-0.5, 0.8), 2), False
        ))

    # Ingest Fraud Transactions
    # Ring 1 (Device Sharing) Transactions: 25 rapid transactions from 7 accounts via DEV_FRAUD_RING1
    logger.info("Injecting Fraud Ring 1 (Device Sharing)...")
    ring1_time = now - datetime.timedelta(hours=1)
    for idx, u in enumerate(ring1_users * 4):
        tx_id = f"TXN_RING1_{idx:03d}"
        amt = round(random.uniform(45000, 98000), 2)
        tx_time = ring1_time + datetime.timedelta(minutes=idx * 2)
        tx_list.append((
            tx_id, u, "DEV_FRAUD_RING1", "185.220.101.44", random.choice(mch_ids), amt, "INR", tx_time, "COMPLETED", idx + 3, round(random.uniform(2.5, 5.0), 2), True
        ))

    # Ring 2 (IP Velocity Proxy) Transactions: 30 transactions from 8 accounts via 198.51.100.99
    logger.info("Injecting Fraud Ring 2 (IP Velocity Proxy)...")
    ring2_time = now - datetime.timedelta(hours=3)
    for idx, u in enumerate(ring2_users * 4):
        tx_id = f"TXN_RING2_{idx:03d}"
        d_id = f"DEV_RING2_{u}"
        amt = round(random.uniform(60000, 120000), 2)
        tx_time = ring2_time + datetime.timedelta(minutes=idx * 3)
        tx_list.append((
            tx_id, u, d_id, "198.51.100.99", "MCH_042", amt, "INR", tx_time, "COMPLETED", idx + 2, round(random.uniform(3.0, 6.0), 2), True
        ))

    # Ring 3 (Carding Attack) Transactions: 15 micro txns in < 3 minutes
    logger.info("Injecting Fraud Ring 3 (Carding Micro-Transactions)...")
    ring3_time = now - datetime.timedelta(minutes=25)
    for idx in range(1, 16):
        tx_id = f"TXN_CARDER_{idx:03d}"
        amt = round(random.uniform(10, 99), 2)
        tx_time = ring3_time + datetime.timedelta(seconds=idx * 10)
        tx_list.append((
            tx_id, ring3_user, "DEV_CARDER_X", "203.0.113.50", random.choice(mch_ids), amt, "INR", tx_time, "COMPLETED", idx, round(random.uniform(1.2, 4.0), 2), True
        ))

    # Ring 4 (Collusion) Transactions: 15 large transactions to MCH_SUSPICIOUS_99
    logger.info("Injecting Fraud Ring 4 (Merchant Collusion)...")
    ring4_time = now - datetime.timedelta(hours=6)
    for idx, u in enumerate(ring4_users * 3):
        tx_id = f"TXN_COLLUSION_{idx:03d}"
        d_id = f"DEV_RING4_{u}"
        amt = round(random.uniform(80000, 150000), 2)
        tx_time = ring4_time + datetime.timedelta(minutes=idx * 5)
        tx_list.append((
            tx_id, u, d_id, ring4_ip[u], "MCH_SUSPICIOUS_99", amt, "INR", tx_time, "COMPLETED", 2, round(random.uniform(2.0, 4.5), 2), True
        ))

    # Insert into Database
    cursor.executemany("""
        INSERT INTO transactions 
        (transaction_id, user_id, device_id, ip_address, merchant_id, amount, currency, timestamp, status, velocity_1h, amount_zscore_prior, is_fraud_ground_truth)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, tx_list)

    conn.commit()
    conn.close()

    logger.info(f"Synthetic Data Generation Complete! Generated {len(tx_list)} total transactions.")
    return len(tx_list)

if __name__ == "__main__":
    generate_dataset()
