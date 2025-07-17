import datetime
from zoneinfo import ZoneInfo

def minutes_since_930am_est():
    est = ZoneInfo("America/New_York")
    now = datetime.datetime.now(est)
    today_930 = now.replace(hour=9, minute=30, second=0, microsecond=0)
    delta = now - today_930
    total_seconds = delta.total_seconds()
    minutes = total_seconds // 60  # Floor division to get completed minutes
    return max(0, int(minutes))

def minutes_since_xx30_est():
    est = ZoneInfo("America/New_York")
    now = datetime.datetime.now(est)
    target_time = now.replace(hour=now.hour, minute=30, second=0, microsecond=0)
    if target_time > now:
        # Calculate hours to subtract to bring target_time into the past
        target_time = target_time.replace(hour=target_time.hour - 1)
    delta = now - target_time
    total_seconds = delta.total_seconds()
    minutes = total_seconds // 60  # Floor division to get completed minutes
    return max(0, int(minutes))