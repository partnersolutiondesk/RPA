from pyhive import hive
import subprocess
import sys
import shutil


# =========================================================
# KERBEROS AUTHENTICATION
# =========================================================

def check_kinit_available():
    """Check if kinit is installed and accessible on this machine."""
    if shutil.which("kinit") is None:
        raise EnvironmentError(
            "kinit not found. Please install MIT Kerberos for Windows "
            "(or ensure it's added to PATH) before running this script."
        )


def kinit(principal, keytab_path=None, password=None):
    """
    Obtain a Kerberos ticket before connecting to Hive.
    Exactly one of keytab_path or password must be provided.

    Parameters:
        principal (str): Kerberos principal, e.g. 'username@example.COM'
                          or a service principal 'hive/host@REALM'
        keytab_path (str or None): path to .keytab file (recommended for automation)
        password (str or None): password for interactive kinit (not recommended for bots,
                                 but supported if keytab is unavailable)
    """
    check_kinit_available()

    if keytab_path and password:
        raise ValueError("Provide either keytab_path OR password, not both.")
    if not keytab_path and not password:
        raise ValueError("You must provide either keytab_path or password to authenticate.")

    try:
        if keytab_path:
            cmd = ["kinit", "-kt", keytab_path, principal]
            result = subprocess.run(cmd, capture_output=True, text=True)
        else:
            cmd = ["kinit", principal]
            result = subprocess.run(cmd, input=f"{password}\n", capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"kinit failed: {result.stderr.strip()}")

        print(f"[OK] Kerberos ticket obtained for '{principal}'")

    except FileNotFoundError:
        raise EnvironmentError("kinit command not found. Is MIT Kerberos installed and in PATH?")
    except Exception as e:
        print(f"[ERROR] Kerberos authentication failed: {e}")
        raise


# =========================================================
# HIVE CONNECTION
# =========================================================

def connect(host, port, database, kerberos_service_name='hive'):
    """
    Establish a Kerberos-authenticated connection to Hive.
    Assumes a valid Kerberos ticket is already active (via kinit()).
    """
    try:
        conn = hive.Connection(
            host=host,
            port=port,
            database=database,
            auth='KERBEROS',
            kerberos_service_name=kerberos_service_name
        )
        cursor = conn.cursor()
        cursor.execute("SET hive.support.concurrency=true")
        cursor.execute("SET hive.txn.manager=org.apache.hadoop.hive.ql.lockmgr.DbTxnManager")
        print(f"[OK] Connected to Hive at {host}:{port}, database='{database}'")
        return conn, cursor
    except Exception as e:
        print(f"[ERROR] Connection to Hive failed: {e}")
        raise


def close_connection(conn):
    """Close the Hive connection safely."""
    try:
        conn.close()
        print("[OK] Connection closed.")
    except Exception as e:
        print(f"[WARNING] Error while closing connection: {e}")


# =========================================================
# CRUD OPERATIONS
# =========================================================

def create_table(cursor, table_name, columns, bucket_column, num_buckets=2):
    """
    OPTIONAL: Only call this if you need to create a brand-new table.
    If the table already exists in the customer's database, do NOT call
    this function — just use TABLE_NAME directly with select_rows/insert_row/etc.

    columns (dict): column_name -> data_type, e.g. {'id': 'INT', 'name': 'STRING'}
    """
    try:
        columns_sql = ", ".join([f"{col} {dtype}" for col, dtype in columns.items()])
        query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
              {columns_sql}
            )
            CLUSTERED BY ({bucket_column}) INTO {num_buckets} BUCKETS
            STORED AS ORC
            TBLPROPERTIES ('transactional'='true')
        """
        cursor.execute(query)
        print(f"[OK] Table '{table_name}' is ready.")
    except Exception as e:
        print(f"[ERROR] create_table failed: {e}")
        raise


def insert_row(cursor, table_name, values):
    """
    Insert a row into a table.
    values (list): values in column order, e.g. [1, 'Alice', 50000]
    """
    try:
        formatted_values = ", ".join(
            f"'{v}'" if isinstance(v, str) else str(v) for v in values
        )
        query = f"INSERT INTO {table_name} VALUES ({formatted_values})"
        cursor.execute(query)
        print(f"[OK] Inserted into '{table_name}': {values}")
    except Exception as e:
        print(f"[ERROR] insert_row failed: {e}")
        raise


def select_rows(cursor, table_name, where_clause=None):
    """
    Select rows from a table. where_clause is optional, e.g. "id = 2"
    """
    try:
        query = f"SELECT * FROM {table_name}"
        if where_clause:
            query += f" WHERE {where_clause}"
        cursor.execute(query)
        rows = cursor.fetchall()
        for row in rows:
            print(row)
        return rows
    except Exception as e:
        print(f"[ERROR] select_rows failed: {e}")
        raise


def update_row(cursor, table_name, set_clause, where_clause):
    """
    Update rows in a table.
    set_clause: e.g. "salary = 70000"
    where_clause: e.g. "id = 4"
    """
    try:
        query = f"UPDATE {table_name} SET {set_clause} WHERE {where_clause}"
        cursor.execute(query)
        print(f"[OK] Updated '{table_name}' SET {set_clause} WHERE {where_clause}")
    except Exception as e:
        print(f"[ERROR] update_row failed: {e}")
        raise


def delete_row(cursor, table_name, where_clause):
    """
    Delete rows from a table.
    where_clause: e.g. "id = 4"
    """
    try:
        query = f"DELETE FROM {table_name} WHERE {where_clause}"
        cursor.execute(query)
        print(f"[OK] Deleted from '{table_name}' WHERE {where_clause}")
    except Exception as e:
        print(f"[ERROR] delete_row failed: {e}")
        raise


# =========================================================
# MAIN - CUSTOMER CONFIGURATION
# =========================================================

def main():
    # ---------------------------------------------------------------
    # EDIT ONLY THIS SECTION — everything you need is here.
    # ---------------------------------------------------------------

    # --- Kerberos authentication settings ---
    PRINCIPAL = "<your_username>@<YOUR_REALM>"        # your Kerberos principal
    KEYTAB_PATH = None                      # e.g. r"C:\path\to\your.keytab"  -- set to None if using password
    PASSWORD = None                         # e.g. "your_password"           -- set to None if using keytab

    # --- Hive connection settings ---
    HOST = "<YOUR_HIVE_HOST_IP>"                      # e.g. "10.4.2022.111"
    PORT = "<YOUR_HIVE_PORT>"                         # e.g. 10000
    DATABASE = "<YOUR_DATABASE_NAME>"                 # e.g. "db"
    KERBEROS_SERVICE_NAME = "<YOUR_KERBEROS_SERVICE_NAME>"          # usually 'hive', confirm with DB team if unsure

    # --- Table name ---
    # If the table ALREADY EXISTS in the customer's database, just set its
    # name here — do NOT call create_table() below.
    # Only call create_table() if you specifically need to create a NEW table.
    TABLE_NAME = "<YOUR_TABLE_NAME>"                # replace with an existing table name if not creating one

    # ---------------------------------------------------------------
    # DO NOT EDIT BELOW THIS LINE
    # ---------------------------------------------------------------

    # Validate that exactly one auth method is provided
    if not KEYTAB_PATH and not PASSWORD:
        print("[ERROR] You must set either KEYTAB_PATH or PASSWORD in the config section above.")
        sys.exit(1)
    if KEYTAB_PATH and PASSWORD:
        print("[ERROR] Set only ONE of KEYTAB_PATH or PASSWORD, not both.")
        sys.exit(1)

    conn = None
    try:
        # Step 1: Kerberos ticket
        kinit(PRINCIPAL, keytab_path=KEYTAB_PATH, password=PASSWORD)

        # Step 2: Connect to Hive
        conn, cursor = connect(HOST, PORT, DATABASE, kerberos_service_name=KERBEROS_SERVICE_NAME)

        # Step 3: Operations — edit/uncomment whichever you need.

        # OPTIONAL — only uncomment if the table does NOT already exist:
        # create_table(cursor, TABLE_NAME, {"id": "INT", "name": "STRING", "salary": "DOUBLE"}, bucket_column="id")

        print("\n--- Select ---")
        select_rows(cursor, TABLE_NAME)

        # insert_row(cursor, TABLE_NAME, [7, "Grace", 58000])
        # update_row(cursor, TABLE_NAME, set_clause="salary = 60000", where_clause="id = 7")
        # delete_row(cursor, TABLE_NAME, where_clause="id = 7")

    except Exception as e:
        print(f"\n[FATAL] Script stopped due to an error: {e}")
        sys.exit(1)

    finally:
        if conn:
            close_connection(conn)


if __name__ == "__main__":
    main()