"""
WRDS connection and raw data extraction.

Prompts for WRDS username interactively. The wrds library caches
credentials in ~/.pgpass after first use.
"""

import wrds
import pandas as pd


def get_connection() -> wrds.Connection:
    """Open a WRDS connection (prompts for credentials if not cached)."""
    return wrds.Connection()


def fetch_sp500_membership(conn: wrds.Connection) -> pd.DataFrame:
    """
    Fetch full S&P 500 historical membership from crsp.dsp500list.

    Returns
    -------
    pd.DataFrame
        permno : int
        start  : datetime — date stock entered S&P 500
        ending : datetime — date stock left (NaT if still active)
    """
    query = """
        SELECT permno, start, ending
        FROM crsp.dsp500list
        ORDER BY permno, start
    """
    df = conn.raw_sql(query)
    df["start"] = pd.to_datetime(df["start"])
    df["ending"] = pd.to_datetime(df["ending"])
    df["permno"] = df["permno"].astype(int)
    return df


def fetch_daily_stock_data(
    conn: wrds.Connection, start_date: str, end_date: str
) -> pd.DataFrame:
    """
    Fetch CRSP daily stock file (dsf) for all securities in date range.

    Parameters
    ----------
    start_date, end_date : str
        Format 'YYYY-MM-DD'.

    Returns
    -------
    pd.DataFrame
        permno, date, ret, prc, shrout, cfacpr, cfacshr
    """
    query = f"""
        SELECT permno, date, ret, prc, shrout, cfacpr, cfacshr
        FROM crsp.dsf
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY permno, date
    """
    df = conn.raw_sql(query)
    df["date"] = pd.to_datetime(df["date"])
    df["permno"] = df["permno"].astype(int)
    return df


def fetch_delisting_returns(
    conn: wrds.Connection, start_date: str, end_date: str
) -> pd.DataFrame:
    """
    Fetch CRSP delisting returns for survivorship-bias correction.

    Parameters
    ----------
    start_date, end_date : str
        Format 'YYYY-MM-DD'.

    Returns
    -------
    pd.DataFrame
        permno, dlstdt, dlret, dlstcd
    """
    query = f"""
        SELECT permno, dlstdt, dlret, dlstcd
        FROM crsp.dsedelist
        WHERE dlstdt BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY permno, dlstdt
    """
    df = conn.raw_sql(query)
    df["dlstdt"] = pd.to_datetime(df["dlstdt"])
    df["permno"] = df["permno"].astype(int)
    return df
