import csv
import similarity_part
import prediction_part
import testing

def remove_last_t_elements(csv_path, t):
    with open(csv_path, newline='', encoding="utf-8") as f:
        reader = list(csv.reader(f))
    
    if t <= 0:
        return []
    if t > len(reader):
        removed = reader
        remaining = []
    else:
        removed = reader[-t:]
        remaining = reader[:-t]

    with open(csv_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(remaining)

    return removed


def readd_elements(csv_path, rows):
    if not rows:
        return

    with open(csv_path, "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

def main():
    
    experiment_name = "test"
    # For backtesting purposes, remove last 21 elements from the CSV
    csv_path = "data/convertcsv.csv"
    t = 21
    removed_rows = remove_last_t_elements(csv_path, t)
    print(f"Removed {len(removed_rows)} rows from {csv_path}.")
    
    try:
        # Run the similarity part to prepare data
        similarity_part.main(experiment_name)
        
        # Run the prediction part to get forecasts
        prediction_part.main(experiment_name)
        
        # After prediction, re-add the removed elements back to the CSV
        readd_elements(csv_path, removed_rows)
        
        # And build the final output comparison CSV
        testing.main(experiment_name)
    
    except Exception as e:
        print(f"An error occurred: {e}")
        # Ensure that we always re-add the removed rows even if an error occurs
        readd_elements(csv_path, removed_rows)
        print(f"Re-added {len(removed_rows)} rows back to {csv_path}.")
    
    
if __name__ == "__main__":
    main()