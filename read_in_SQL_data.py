import sqlite3

def g_string(g):
    if abs(g - 1) < 10 ** -10:
        retrun "1."
    else:
        return f"{g:.1f}"

conn = sqlite3.connect(f"cosmhal-scalar-hbor-su{N}-L{L}_g{g_string(g)}_m2{m}_or{OR}_database.0.db")

cur = conn.cursor()

database = cur.execute('select * from Observables')

data = database.getall()

