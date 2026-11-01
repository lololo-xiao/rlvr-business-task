"""Build the SQLite business database with fixed, hand-designed records.

Each order is crafted so that ignoring one hidden constraint produces a
different (wrong) answer. Run: python -m rlvr_task.data.build_db
"""
import sqlite3
from pathlib import Path

from rlvr_task.config import DB_PATH

SCHEMA = """
CREATE TABLE customers (
    customer_id TEXT PRIMARY KEY, name TEXT, tier TEXT, phone TEXT, registered_at TEXT
);
CREATE TABLE orders (
    order_id TEXT PRIMARY KEY, customer_id TEXT, order_date TEXT, order_total REAL,
    status TEXT, remark TEXT
);
CREATE TABLE order_items (
    item_id TEXT PRIMARY KEY, order_id TEXT, name TEXT, category TEXT,
    unit_price REAL, qty INTEGER, returnable INTEGER
);
CREATE TABLE deliveries (
    order_id TEXT PRIMARY KEY, carrier TEXT, tracking_no TEXT, shipped_at TEXT,
    delivered_at TEXT, return_tracking_no TEXT, return_status TEXT
);
CREATE TABLE attachments (
    attachment_id TEXT PRIMARY KEY, order_id TEXT, item_id TEXT, file_name TEXT,
    damage_type TEXT, damaged_qty INTEGER, taken_at TEXT, uploaded_at TEXT
);
CREATE TABLE addresses (
    order_id TEXT PRIMARY KEY, customer_address TEXT, merchant_address TEXT
);
CREATE TABLE refunds (
    refund_id TEXT PRIMARY KEY, order_id TEXT, item_id TEXT, amount REAL, created_at TEXT
);
CREATE TABLE notifications (
    order_id TEXT, message TEXT
);
"""

CUSTOMERS = [
    ("C01", "林一", "silver", "138-0000-0001", "2024-03-02"),
    ("C02", "周二", "bronze", "138-0000-0002", "2026-08-15"),
    ("C03", "陈三", "gold", "138-0000-0003", "2021-06-30"),
    ("C04", "吴四", "silver", "138-0000-0004", "2025-01-11"),
    ("C05", "郑五", "gold", "138-0000-0005", "2022-09-09"),
]

# order_total = sum of all items (a distractor: refunds are per damaged item)
ORDERS = [
    ("O1001", "C01", "2026-10-12", 358.0, "delivered", "请放快递柜"),
    ("O1002", "C02", "2026-10-20", 199.0, "delivered", ""),
    ("O1003", "C04", "2026-10-25", 159.0, "delivered", "工作日送货"),
    ("O1004", "C01", "2026-10-22", 719.0, "delivered", ""),
    ("O1005", "C03", "2026-10-24", 1139.0, "delivered", "礼品包装"),
    ("O1006", "C05", "2026-10-26", 4999.0, "delivered", ""),
    ("O1007", "C03", "2026-10-05", 459.0, "delivered", ""),
    ("O1008", "C02", "2026-10-27", 520.0, "delivered", ""),
    ("O1009", "C04", "2026-10-27", 270.0, "delivered", "易碎品"),
    ("O0901", "C01", "2026-09-01", 89.0, "completed", ""),
]

ORDER_ITEMS = [
    ("I1001A", "O1001", "蓝牙耳机", "audio", 299.0, 1, 1),
    ("I1001B", "O1001", "手机壳", "accessory", 59.0, 1, 1),
    ("I1002A", "O1002", "电动牙刷", "personal_care", 199.0, 1, 1),
    ("I1003A", "O1003", "护眼台灯", "home", 159.0, 1, 1),
    ("I1004A", "O1004", "机械键盘", "computer_peripheral", 680.0, 1, 1),
    ("I1004B", "O1004", "鼠标垫", "accessory", 39.0, 1, 1),
    ("I1005A", "O1005", "咖啡机", "kitchen", 899.0, 1, 1),
    ("I1005B", "O1005", "咖啡豆", "food", 120.0, 2, 1),
    ("I1006A", "O1006", "笔记本电脑", "computer", 4999.0, 1, 0),
    ("I1007A", "O1007", "空气炸锅", "kitchen", 459.0, 1, 1),
    ("I1008A", "O1008", "投影仪", "electronics", 520.0, 1, 1),
    ("I1009A", "O1009", "玻璃杯", "kitchen", 60.0, 4, 1),
    ("I1009B", "O1009", "餐垫", "home", 30.0, 1, 1),
    ("I0901A", "O0901", "数据线", "accessory", 89.0, 1, 1),
]

DELIVERIES = [
    # O1001: 20 days since order but only 12 since delivery -> within silver 15d
    ("O1001", "顺丰", "SF100001", "2026-10-13 09:00", "2026-10-20 10:00", "SFR90001", "arrived_at_merchant"),
    # O1002: delivered 9 days ago, bronze window 7d -> expired
    ("O1002", "中通", "ZT100002", "2026-10-21 09:00", "2026-10-23 15:00", "ZTR90002", "arrived_at_merchant"),
    ("O1003", "京东", "JD100003", "2026-10-26 09:00", "2026-10-28 09:00", "JDR90003", "arrived_at_merchant"),
    ("O1004", "顺丰", "SF100004", "2026-10-23 09:00", "2026-10-27 11:00", "SFR90004", "arrived_at_merchant"),
    # O1005: return parcel still in transit
    ("O1005", "圆通", "YT100005", "2026-10-25 09:00", "2026-10-28 18:00", "YTR90005", "in_transit"),
    ("O1006", "京东", "JD100006", "2026-10-27 09:00", "2026-10-29 10:00", "JDR90006", "arrived_at_merchant"),
    # O1007: gold, delivered 20 days ago -> still within gold 30d
    ("O1007", "顺丰", "SF100007", "2026-10-06 09:00", "2026-10-12 10:00", "SFR90007", "arrived_at_merchant"),
    # O1008: bronze threshold 500 < 520 -> escalate
    ("O1008", "中通", "ZT100008", "2026-10-28 09:00", "2026-10-29 12:00", "ZTR90008", "arrived_at_merchant"),
    # O1009: 2 of 4 glasses broken -> refund 2 x 60, not the whole line / order
    ("O1009", "京东", "JD100009", "2026-10-28 09:00", "2026-10-30 09:00", "JDR90009", "arrived_at_merchant"),
    ("O0901", "顺丰", "SF090001", "2026-09-02 09:00", "2026-09-04 10:00", None, None),
]

ATTACHMENTS = [
    ("A1", "O1001", "I1001A", "earbuds_broken.jpg", "外壳破裂", 1, "2026-10-20 20:30", "2026-10-21 08:00"),
    ("A2", "O1002", "I1002A", "brush.jpg", "无法开机", 1, "2026-10-23 18:00", "2026-10-30 09:00"),
    # O1003: photo taken 54h after delivery -> invalid evidence
    ("A3", "O1003", "I1003A", "lamp.jpg", "灯罩碎裂", 1, "2026-10-30 15:00", "2026-10-30 16:00"),
    ("A4", "O1004", "I1004A", "keyboard.jpg", "按键脱落", 1, "2026-10-27 19:00", "2026-10-27 19:10"),
    ("A5", "O1005", "I1005A", "coffee.jpg", "水箱开裂", 1, "2026-10-29 08:00", "2026-10-29 08:30"),
    ("A6", "O1006", "I1006A", "laptop.jpg", "屏幕划痕", 1, "2026-10-29 12:00", "2026-10-29 12:05"),
    ("A7", "O1007", "I1007A", "fryer.jpg", "炸篮变形", 1, "2026-10-12 15:00", "2026-10-13 09:00"),
    ("A8", "O1008", "I1008A", "projector.jpg", "镜头碎裂", 1, "2026-10-29 20:00", "2026-10-29 20:10"),
    ("A9", "O1009", "I1009A", "glasses.jpg", "两只碎裂", 2, "2026-10-30 12:00", "2026-10-31 10:00"),
]

ADDRESSES = [
    (oid, f"{cid} 的收货地址（模拟）", "杭州市余杭区仓储中心 1 号库")
    for oid, cid, *_ in ORDERS
]


def build(db_path: Path = DB_PATH) -> Path:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.executemany("INSERT INTO customers VALUES (?,?,?,?,?)", CUSTOMERS)
        conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?)", ORDERS)
        conn.executemany("INSERT INTO order_items VALUES (?,?,?,?,?,?,?)", ORDER_ITEMS)
        conn.executemany("INSERT INTO deliveries VALUES (?,?,?,?,?,?,?)", DELIVERIES)
        conn.executemany("INSERT INTO attachments VALUES (?,?,?,?,?,?,?,?)", ATTACHMENTS)
        conn.executemany("INSERT INTO addresses VALUES (?,?,?)", ADDRESSES)
        conn.commit()
    finally:
        conn.close()
    return db_path


if __name__ == "__main__":
    print(f"built {build()}")
