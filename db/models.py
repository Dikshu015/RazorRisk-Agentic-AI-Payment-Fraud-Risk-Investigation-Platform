from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.sql import func
from db.database import Base

class User(Base):
    __tablename__ = "users"
    user_id = Column(String(50), primary_key=True)
    name = Column(String(100))
    email = Column(String(100))
    created_at = Column(DateTime, default=func.now())
    account_status = Column(String(20), default="ACTIVE")

class Device(Base):
    __tablename__ = "devices"
    device_id = Column(String(50), primary_key=True)
    device_type = Column(String(50))
    os = Column(String(50))
    is_vpn_proxy = Column(Boolean, default=False)
    first_seen = Column(DateTime, default=func.now())

class IPAddress(Base):
    __tablename__ = "ip_addresses"
    ip_address = Column(String(45), primary_key=True)
    country = Column(String(50))
    city = Column(String(50))
    isp = Column(String(100))
    is_suspicious_proxy = Column(Boolean, default=False)

class Merchant(Base):
    __tablename__ = "merchants"
    merchant_id = Column(String(50), primary_key=True)
    name = Column(String(100))
    category = Column(String(50))
    fraud_rate = Column(Float, default=0.01)
    created_at = Column(DateTime, default=func.now())

class Transaction(Base):
    __tablename__ = "transactions"
    transaction_id = Column(String(50), primary_key=True)
    user_id = Column(String(50), ForeignKey("users.user_id"))
    device_id = Column(String(50), ForeignKey("devices.device_id"))
    ip_address = Column(String(45), ForeignKey("ip_addresses.ip_address"))
    merchant_id = Column(String(50), ForeignKey("merchants.merchant_id"))
    amount = Column(Float, nullable=False)
    currency = Column(String(10), default="INR")
    timestamp = Column(DateTime, default=func.now())
    status = Column(String(20), default="COMPLETED")
    velocity_1h = Column(Integer, default=1)
    amount_zscore_prior = Column(Float, default=0.0)
    is_fraud_ground_truth = Column(Boolean, default=False)

class RiskScoreRecord(Base):
    __tablename__ = "risk_scores"
    scoring_id = Column(String(50), primary_key=True)
    transaction_id = Column(String(50), ForeignKey("transactions.transaction_id"))
    risk_score = Column(Float, nullable=False)
    tabular_score = Column(Float, nullable=False)
    gnn_score = Column(Float, nullable=False)
    velocity_multiplier = Column(Float, default=1.0)
    risk_tier = Column(String(20), nullable=False)
    decision = Column(String(30), nullable=False)
    scored_at = Column(DateTime, default=func.now())

class InvestigationReportRecord(Base):
    __tablename__ = "investigation_reports"
    investigation_id = Column(String(50), primary_key=True)
    transaction_id = Column(String(50), ForeignKey("transactions.transaction_id"))
    risk_score = Column(Float, nullable=False)
    evidence_json = Column(Text, nullable=False)
    fraud_hypothesis = Column(Text, nullable=False)
    recommended_action = Column(String(50), nullable=False)
    summary_report = Column(Text, nullable=False)
    created_at = Column(DateTime, default=func.now())

class SystemLogRecord(Base):
    __tablename__ = "system_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    module = Column(String(50))
    level = Column(String(20))
    message = Column(Text)
    created_at = Column(DateTime, default=func.now())
