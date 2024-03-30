# for each of the .csv file in ./data, read the first line and store in ./data/first_line.csv
import csv, os

for file in os.listdir("./data/val"):
    with open(f"./data/val/{file}", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            with open(f"./data/first_line.csv", "a", encoding="utf-8") as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow(row)
            break