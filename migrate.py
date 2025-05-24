import sqlite3

# Connect to your actual database file
conn = sqlite3.connect('database.db')
cursor = conn.cursor()

# Step 1: Add the column without a default value
try:
    cursor.execute("ALTER TABLE predictions ADD COLUMN created_at TIMESTAMP")
    print("Column 'created_at' added.")
except sqlite3.OperationalError as e:
    print("Skipping column addition (it may already exist):", e)

# Step 2: Set existing rows to current timestamp
cursor.execute("UPDATE predictions SET created_at = CURRENT_TIMESTAMP WHERE created_at IS NULL")
print("Existing rows updated with CURRENT_TIMESTAMP.")

# Finish up
conn.commit()
conn.close()
print("Migration complete.")
