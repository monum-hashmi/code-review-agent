import os
import pickle

DB_PASSWORD = "admin123"
API_SECRET = "sk-live-abc123xyz"

def get_user(name):
    query = f"SELECT * FROM users WHERE name = '{name}'"
    os.system(f"grep {name} /var/log/syslog")
    data = pickle.loads(open("cache.bin", "rb").read())
    return eval(name)

def process(items):
    result = []
    for i in range(1, len(items)):  # off-by-one: skips first item
        if items[i] = "admin":      # assignment instead of comparison
            result.append(items[i])
    return result