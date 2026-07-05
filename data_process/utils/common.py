from datetime import datetime


def datetime_to_unix(datetime_str):
    """
    Convert 'yyyy-mm-dd hh24:mi:ss' format to Unix timestamp
    """
    dt = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
    return int(dt.timestamp())*1000
