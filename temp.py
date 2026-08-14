import pandas as pd
import os
import shutil


# ============================================================
# SETTINGS
# ============================================================

input_file = (
    r"C:\Users\chskc\Desktop\FirstSource\Intelligent-Customer-Signal-Detector\datasets\customer_data.csv"

)
customer_id_col = "customer_id"

churn_col = "churn_score_1"

source_chat_folder = (
    r"C:\Users\chskc\Desktop\FirstSource\Intelligent-Customer-Signal-Detector\datasets\customer_chat_data"
)

# SINGLE OUTPUT PATH
output_folder = (
    r"C:\Users\chskc\Desktop\FirstSource\output3"
)

# Number of batches
total_batches = 100

# Customers in each batch
batch_size = 10

# Random seed
random_seed = 42


# ============================================================
# CHURN DISTRIBUTION
# ============================================================

# <50 must be between 40% and 70%
min_less_50_ratio = 0.40
max_less_50_ratio = 0.70


# ============================================================
# LOAD CSV
# ============================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_csv(input_file)

print(f"Original rows: {len(df):,}")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    customer_id_col,
    churn_col
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# CLEAN CUSTOMER IDs
# ============================================================

df[customer_id_col] = (
    df[customer_id_col]
    .astype(str)
    .str.strip()
)


# ============================================================
# REMOVE EMPTY CUSTOMER IDs
# ============================================================

df = df[
    (df[customer_id_col] != "") &
    (df[customer_id_col].str.lower() != "nan")
].copy()


# ============================================================
# REMOVE DUPLICATE CUSTOMER IDs
# ============================================================

before = len(df)

df = df.drop_duplicates(
    subset=[customer_id_col],
    keep="first"
).reset_index(drop=True)

after = len(df)

print(
    f"Duplicate customer IDs removed: "
    f"{before - after:,}"
)

print(
    f"Unique customers after cleaning: "
    f"{after:,}"
)


# ============================================================
# CONVERT churn_score_1 TO NUMERIC
# ============================================================

df[churn_col] = pd.to_numeric(
    df[churn_col],
    errors="coerce"
)


# ============================================================
# REMOVE INVALID CHURN VALUES
# ============================================================

invalid_count = df[churn_col].isna().sum()

if invalid_count > 0:

    print(
        f"WARNING: {invalid_count:,} invalid "
        f"churn_score_1 values removed."
    )

    df = df[
        df[churn_col].notna()
    ].copy()


# ============================================================
# CREATE TWO GROUPS
# ============================================================

df_less_50 = df[
    df[churn_col] < 50
].copy()

df_more_50 = df[
    df[churn_col] > 50
].copy()

df_equal_50 = df[
    df[churn_col] == 50
].copy()


# ============================================================
# PRINT DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("CHURN GROUP DISTRIBUTION")
print("=" * 70)

print(
    f"churn_score_1 < 50 : {len(df_less_50):,}"
)

print(
    f"churn_score_1 = 50 : {len(df_equal_50):,}"
)

print(
    f"churn_score_1 > 50 : {len(df_more_50):,}"
)

print("=" * 70)


# ============================================================
# BATCH REQUIREMENT
# ============================================================

required_customers = (
    total_batches * batch_size
)

print()
print("=" * 70)
print("BATCH REQUIREMENT")
print("=" * 70)

print(
    f"Number of batches : {total_batches}"
)

print(
    f"Batch size        : {batch_size}"
)

print(
    f"Total required    : {required_customers:,}"
)

print("=" * 70)


# ============================================================
# CALCULATE BATCH DISTRIBUTION
# ============================================================

min_less_50 = int(
    batch_size * min_less_50_ratio
)

max_less_50 = int(
    batch_size * max_less_50_ratio
)


print()
print("=" * 70)
print("ALLOWED DISTRIBUTION")
print("=" * 70)

print(
    f"Minimum <50 per batch: {min_less_50}"
)

print(
    f"Maximum <50 per batch: {max_less_50}"
)

print("=" * 70)


# ============================================================
# CHECK ENOUGH CUSTOMERS
# ============================================================

if len(df_less_50) < (
    total_batches * min_less_50
):

    raise ValueError(
        "Not enough customers with "
        "churn_score_1 < 50."
    )


if len(df_more_50) < (
    total_batches *
    (batch_size - max_less_50)
):

    raise ValueError(
        "Not enough customers with "
        "churn_score_1 > 50."
    )


# ============================================================
# IMPORTANT:
#
# USE EXACTLY 50% / 50%
#
# Since your data has:
#
#   504 <50
#   504 >50
#
# and we need:
#
#   1000 total
#
# we use:
#
#   500 <50
#   500 >50
#
# This gives every batch:
#
#   50 <50
#   50 >50
#
# = 50% / 50%
#
# which satisfies 40%-70%.
# ============================================================

less_50_per_batch = batch_size // 2

more_50_per_batch = (
    batch_size - less_50_per_batch
)


print()
print("=" * 70)
print("ACTUAL BATCH DISTRIBUTION")
print("=" * 70)

print(
    f"<50 per batch: {less_50_per_batch}"
)

print(
    f">50 per batch: {more_50_per_batch}"
)

print(
    f"<50 percentage: "
    f"{less_50_per_batch / batch_size * 100:.1f}%"
)

print(
    f">50 percentage: "
    f"{more_50_per_batch / batch_size * 100:.1f}%"
)

print("=" * 70)


# ============================================================
# SHUFFLE BOTH GROUPS
# ============================================================

df_less_50 = df_less_50.sample(
    frac=1,
    random_state=random_seed
).reset_index(drop=True)


df_more_50 = df_more_50.sample(
    frac=1,
    random_state=random_seed + 1
).reset_index(drop=True)


# ============================================================
# CHECK TOTAL CUSTOMERS REQUIRED FROM EACH GROUP
# ============================================================

required_less_50 = (
    total_batches *
    less_50_per_batch
)

required_more_50 = (
    total_batches *
    more_50_per_batch
)


if len(df_less_50) < required_less_50:

    raise ValueError(
        f"Need {required_less_50} customers "
        f"with <50 but only have "
        f"{len(df_less_50)}."
    )


if len(df_more_50) < required_more_50:

    raise ValueError(
        f"Need {required_more_50} customers "
        f"with >50 but only have "
        f"{len(df_more_50)}."
    )


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(
    output_folder,
    exist_ok=True
)


# ============================================================
# GLOBAL CUSTOMER TRACKING
# ============================================================

used_customer_ids = set()


# ============================================================
# CREATE BATCHES
# ============================================================

for batch_number in range(
    1,
    total_batches + 1
):

    print()
    print("#" * 70)
    print(
        f"CREATING BATCH {batch_number}"
    )
    print("#" * 70)


    # ========================================================
    # CALCULATE POSITIONS
    # ========================================================

    less_start = (
        (batch_number - 1)
        * less_50_per_batch
    )

    less_end = (
        less_start +
        less_50_per_batch
    )


    more_start = (
        (batch_number - 1)
        * more_50_per_batch
    )

    more_end = (
        more_start +
        more_50_per_batch
    )


    # ========================================================
    # GET <50 CUSTOMERS
    # ========================================================

    batch_less_50 = df_less_50.iloc[
        less_start:less_end
    ].copy()


    # ========================================================
    # GET >50 CUSTOMERS
    # ========================================================

    batch_more_50 = df_more_50.iloc[
        more_start:more_end
    ].copy()


    # ========================================================
    # COMBINE
    # ========================================================

    batch_df = pd.concat(
        [
            batch_less_50,
            batch_more_50
        ],
        ignore_index=True
    )


    # ========================================================
    # SHUFFLE BATCH
    # ========================================================

    batch_df = batch_df.sample(
        frac=1,
        random_state=random_seed + batch_number
    ).reset_index(drop=True)


    # ========================================================
    # CHECK BATCH SIZE
    # ========================================================

    if len(batch_df) != batch_size:

        raise ValueError(
            f"Batch {batch_number} has "
            f"{len(batch_df)} rows instead of "
            f"{batch_size}."
        )


    # ========================================================
    # CHECK DUPLICATES INSIDE BATCH
    # ========================================================

    duplicate_inside = (
        batch_df[customer_id_col]
        .duplicated()
        .sum()
    )


    if duplicate_inside > 0:

        raise ValueError(
            f"Duplicate customer found inside "
            f"batch {batch_number}."
        )


    # ========================================================
    # CHECK CROSS-BATCH DUPLICATES
    # ========================================================

    current_ids = set(
        batch_df[customer_id_col]
    )


    duplicates_with_previous = (
        current_ids &
        used_customer_ids
    )


    if duplicates_with_previous:

        raise ValueError(
            f"Cross-batch duplicate detected "
            f"in batch {batch_number}:\n"
            f"{duplicates_with_previous}"
        )


    # Add IDs to global tracking
    used_customer_ids.update(
        current_ids
    )


    # ========================================================
    # VERIFY CHURN DISTRIBUTION
    # ========================================================

    batch_less_count = (
        batch_df[churn_col] < 50
    ).sum()


    batch_more_count = (
        batch_df[churn_col] > 50
    ).sum()


    less_percentage = (
        batch_less_count /
        len(batch_df)
    ) * 100


    more_percentage = (
        batch_more_count /
        len(batch_df)
    ) * 100


    # ========================================================
    # CHECK 40%-70%
    # ========================================================

    if not (
        40 <= less_percentage <= 70
    ):

        raise ValueError(
            f"Batch {batch_number} does not satisfy "
            f"the 40%-70% requirement."
        )


    # ========================================================
    # PRINT BATCH DETAILS
    # ========================================================

    print(
        f"Total customers : {len(batch_df)}"
    )

    print(
        f"<50 customers   : "
        f"{batch_less_count} "
        f"({less_percentage:.1f}%)"
    )

    print(
        f">50 customers   : "
        f"{batch_more_count} "
        f"({more_percentage:.1f}%)"
    )


    # ========================================================
    # CREATE BATCH FOLDER
    # ========================================================

    batch_folder = os.path.join(
        output_folder,
        f"batch{batch_number}"
    )

    os.makedirs(
        batch_folder,
        exist_ok=True
    )


    # ========================================================
    # DROP churn_score_1
    # ========================================================
    #
    # IMPORTANT:
    #
    # We use batch_df above for validation.
    #
    # Only the dataframe being saved has churn_score_1
    # removed.
    # ========================================================

    batch_to_save = batch_df.drop(
        columns=[churn_col]
    ).copy()


    # ========================================================
    # SAVE CSV
    # ========================================================

    csv_path = os.path.join(
        batch_folder,
        f"batch{batch_number}.csv"
    )


    batch_to_save.to_csv(
        csv_path,
        index=False
    )


    print(
        f"CSV saved: {csv_path}"
    )


    # ========================================================
    # COPY TXT FILES
    # ========================================================

    copied_count = 0
    missing_count = 0


    for customer_id in batch_df[
        customer_id_col
    ]:

        txt_filename = (
            f"{customer_id}.txt"
        )


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

            copied_count += 1

        else:

            print(
                f"WARNING: TXT not found: "
                f"{txt_filename}"
            )

            missing_count += 1


    print(
        f"TXT files copied : {copied_count}"
    )

    print(
        f"TXT files missing: {missing_count}"
    )


    print(
        f"Batch {batch_number} completed."
    )


# ============================================================
# FINAL VERIFICATION
# ============================================================

print()
print("=" * 70)
print("FINAL VERIFICATION")
print("=" * 70)


all_batch_ids = []


for batch_number in range(
    1,
    total_batches + 1
):

    csv_path = os.path.join(
        output_folder,
        f"batch{batch_number}",
        f"batch{batch_number}.csv"
    )


    # --------------------------------------------------------
    # Check file exists
    # --------------------------------------------------------

    if not os.path.exists(csv_path):

        raise FileNotFoundError(
            f"Missing batch CSV: {csv_path}"
        )


    # --------------------------------------------------------
    # Read saved CSV
    # --------------------------------------------------------

    batch_check = pd.read_csv(
        csv_path
    )


    # --------------------------------------------------------
    # Check batch size
    # --------------------------------------------------------

    if len(batch_check) != batch_size:

        raise ValueError(
            f"Batch {batch_number} contains "
            f"{len(batch_check)} rows."
        )


    # --------------------------------------------------------
    # Check churn_score_1 was removed
    # --------------------------------------------------------

    if churn_col in batch_check.columns:

        raise ValueError(
            f"{churn_col} still exists in "
            f"batch {batch_number}."
        )


    # --------------------------------------------------------
    # Collect IDs
    # --------------------------------------------------------

    ids = (
        batch_check[
            customer_id_col
        ]
        .astype(str)
        .str.strip()
        .tolist()
    )


    all_batch_ids.extend(ids)


    print(
        f"Batch {batch_number}: "
        f"{len(ids)} customers verified"
    )


# ============================================================
# CHECK CROSS-BATCH DUPLICATES
# ============================================================

all_ids_series = pd.Series(
    all_batch_ids
)


duplicate_ids = (
    all_ids_series[
        all_ids_series.duplicated()
    ]
    .unique()
)


print()


if len(duplicate_ids) == 0:

    print(
        "SUCCESS: No cross-batch duplicates."
    )

else:

    print(
        "ERROR: Duplicate customers found:"
    )

    print(
        duplicate_ids
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

total_saved = len(
    all_batch_ids
)

unique_saved = len(
    set(all_batch_ids)
)


print()
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(
    f"Total batches          : {total_batches}"
)

print(
    f"Customers per batch    : {batch_size}"
)

print(
    f"Total saved customers  : {total_saved:,}"
)

print(
    f"Unique customers       : {unique_saved:,}"
)

print(
    f"Duplicate customers    : "
    f"{total_saved - unique_saved}"
)

print(
    f"Output folder          : "
    f"{output_folder}"
)

print("=" * 70)

print("\nDONE!")



# import pandas as pd
# import os
# import shutil

# # --------------------------------------------------
# # SETTINGS
# # --------------------------------------------------

# input_file = r"C:\Users\chskc\Desktop\FirstSource\Intelligent-Customer-Signal-Detector\datasets\Telco_customer_dashboard.csv"

# customer_id_col = "customer_id"

# source_chat_folder = r"C:\Users\chskc\Desktop\FirstSource\Intelligent-Customer-Signal-Detector\datasets\customer_chat_data"

# output_folder = r"C:\Users\chskc\Desktop\FirstSource\output2"

# total_batches = 2#10
# batch_size = 10 #0

# # Set a number if you want the same random batches every time.
# # Change it to another number for a different random arrangement.
# random_seed = 42


# # --------------------------------------------------
# # LOAD CSV
# # --------------------------------------------------

# df = pd.read_csv(input_file)

# # Clean customer IDs
# df[customer_id_col] = df[customer_id_col].astype(str).str.strip()


# # --------------------------------------------------
# # REMOVE DUPLICATE CUSTOMER IDs
# # --------------------------------------------------

# # This guarantees that the same customer cannot
# # appear more than once in the entire dataset.
# df = df.drop_duplicates(
#     subset=[customer_id_col],
#     keep="first"
# ).reset_index(drop=True)


# # --------------------------------------------------
# # CHECK WHETHER ENOUGH CUSTOMERS EXIST
# # --------------------------------------------------

# required_customers = total_batches * batch_size

# if len(df) < required_customers:
#     raise ValueError(
#         f"Not enough unique customers.\n"
#         f"Required: {required_customers}\n"
#         f"Available: {len(df)}"
#     )


# # --------------------------------------------------
# # RANDOMLY SHUFFLE ALL CUSTOMERS ONCE
# # --------------------------------------------------

# df = df.sample(
#     frac=1,
#     random_state=random_seed
# ).reset_index(drop=True)


# # Take exactly the number of customers needed
# df = df.iloc[:required_customers].copy()


# # --------------------------------------------------
# # CREATE BATCHES
# # --------------------------------------------------

# for batch_number in range(1, total_batches + 1):

#     start = (batch_number - 1) * batch_size
#     end = start + batch_size

#     batch_df = df.iloc[start:end].copy()

#     # Create batch folder
#     batch_folder = os.path.join(
#         output_folder,
#         f"batch{batch_number}"
#     )

#     os.makedirs(batch_folder, exist_ok=True)


#     # --------------------------------------------------
#     # SAVE CSV
#     # --------------------------------------------------

#     csv_path = os.path.join(
#         batch_folder,
#         f"batch{batch_number}.csv"
#     )

#     batch_df.to_csv(
#         csv_path,
#         index=False
#     )


#     # --------------------------------------------------
#     # COPY CUSTOMER TXT FILES
#     # --------------------------------------------------

#     for customer_id in batch_df[customer_id_col]:

#         txt_filename = f"{customer_id}.txt"

#         source_txt = os.path.join(
#             source_chat_folder,
#             txt_filename
#         )

#         destination_txt = os.path.join(
#             batch_folder,
#             txt_filename
#         )

#         if os.path.exists(source_txt):
#             shutil.copy2(
#                 source_txt,
#                 destination_txt
#             )
#         else:
#             print(
#                 f"WARNING: TXT not found: {txt_filename}"
#             )


#     print(
#         f"batch{batch_number}: "
#         f"{len(batch_df)} unique customers created"
#     )


# # --------------------------------------------------
# # FINAL VERIFICATION
# # --------------------------------------------------

# print("\nChecking for cross-batch duplicates...")

# all_batch_ids = []

# for batch_number in range(1, total_batches + 1):

#     csv_path = os.path.join(
#         output_folder,
#         f"batch{batch_number}",
#         f"batch{batch_number}.csv"
#     )

#     batch_check = pd.read_csv(csv_path)

#     all_batch_ids.extend(
#         batch_check[customer_id_col]
#         .astype(str)
#         .str.strip()
#         .tolist()
#     )


# duplicate_ids = pd.Series(all_batch_ids)[
#     pd.Series(all_batch_ids).duplicated()
# ].unique()


# if len(duplicate_ids) == 0:
#     print("SUCCESS: No customer appears in multiple batches.")
# else:
#     print("ERROR: Duplicate customers found:")
#     print(duplicate_ids)


# print("\nDone!")
