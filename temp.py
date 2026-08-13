import pandas as pd
import os
import shutil

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

input_file = r"C:\Users\chskc\Desktop\FirstSource\Intelligent-Customer-Signal-Detector\datasets\Telco_customer_dashboard.csv"

customer_id_col = "customer_id"

source_chat_folder = r"C:\Users\chskc\Desktop\FirstSource\Intelligent-Customer-Signal-Detector\datasets\customer_chat_data"

output_folder = r"C:\Users\chskc\Desktop\FirstSource\Intelligent-Customer-Signal-Detector\datasets\output"

total_batches = 2#10
batch_size = 30 #10

# Set a number if you want the same random batches every time.
# Change it to another number for a different random arrangement.
random_seed = 42


# --------------------------------------------------
# LOAD CSV
# --------------------------------------------------

df = pd.read_csv(input_file)

# Clean customer IDs
df[customer_id_col] = df[customer_id_col].astype(str).str.strip()


# --------------------------------------------------
# REMOVE DUPLICATE CUSTOMER IDs
# --------------------------------------------------

# This guarantees that the same customer cannot
# appear more than once in the entire dataset.
df = df.drop_duplicates(
    subset=[customer_id_col],
    keep="first"
).reset_index(drop=True)


# --------------------------------------------------
# CHECK WHETHER ENOUGH CUSTOMERS EXIST
# --------------------------------------------------

required_customers = total_batches * batch_size

if len(df) < required_customers:
    raise ValueError(
        f"Not enough unique customers.\n"
        f"Required: {required_customers}\n"
        f"Available: {len(df)}"
    )


# --------------------------------------------------
# RANDOMLY SHUFFLE ALL CUSTOMERS ONCE
# --------------------------------------------------

df = df.sample(
    frac=1,
    random_state=random_seed
).reset_index(drop=True)


# Take exactly the number of customers needed
df = df.iloc[:required_customers].copy()


# --------------------------------------------------
# CREATE BATCHES
# --------------------------------------------------

for batch_number in range(1, total_batches + 1):

    start = (batch_number - 1) * batch_size
    end = start + batch_size

    batch_df = df.iloc[start:end].copy()

    # Create batch folder
    batch_folder = os.path.join(
        output_folder,
        f"batch{batch_number}"
    )

    os.makedirs(batch_folder, exist_ok=True)


    # --------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------

    csv_path = os.path.join(
        batch_folder,
        f"batch{batch_number}.csv"
    )

    batch_df.to_csv(
        csv_path,
        index=False
    )


    # --------------------------------------------------
    # COPY CUSTOMER TXT FILES
    # --------------------------------------------------

    for customer_id in batch_df[customer_id_col]:

        txt_filename = f"{customer_id}.txt"

        source_txt = os.path.join(
            source_chat_folder,
            txt_filename
        )

        destination_txt = os.path.join(
            batch_folder,
            txt_filename
        )

        if os.path.exists(source_txt):
            shutil.copy2(
                source_txt,
                destination_txt
            )
        else:
            print(
                f"WARNING: TXT not found: {txt_filename}"
            )


    print(
        f"batch{batch_number}: "
        f"{len(batch_df)} unique customers created"
    )


# --------------------------------------------------
# FINAL VERIFICATION
# --------------------------------------------------

print("\nChecking for cross-batch duplicates...")

all_batch_ids = []

for batch_number in range(1, total_batches + 1):

    csv_path = os.path.join(
        output_folder,
        f"batch{batch_number}",
        f"batch{batch_number}.csv"
    )

    batch_check = pd.read_csv(csv_path)

    all_batch_ids.extend(
        batch_check[customer_id_col]
        .astype(str)
        .str.strip()
        .tolist()
    )


duplicate_ids = pd.Series(all_batch_ids)[
    pd.Series(all_batch_ids).duplicated()
].unique()


if len(duplicate_ids) == 0:
    print("SUCCESS: No customer appears in multiple batches.")
else:
    print("ERROR: Duplicate customers found:")
    print(duplicate_ids)


print("\nDone!")
