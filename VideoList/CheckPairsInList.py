import os

# URL Duplicate Cleaner Script
# Checks for duplicate URLs in a .txt file and removes the second occurrence.

TXT_FILENAME = "urls.txt"

def clean_duplicates(filename):
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(SCRIPT_DIR, TXT_FILENAME)
    try:
        with open(filename, "r", encoding="utf-8") as file:
            lines = file.readlines()

        seen = set()
        cleaned_lines = []
        duplicates_found = False

        for line in lines:
            url = line.strip()
            if url in seen:
                print(f"Duplicate found and removed: {url}")
                duplicates_found = True
                continue
            seen.add(url)
            cleaned_lines.append(url + "\n")

        with open(filename, "w", encoding="utf-8") as file:
            file.writelines(cleaned_lines)

        if not duplicates_found:
            print("No duplicate URLs found.")
        else:
            print("Duplicates have been removed from the file.")

    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    clean_duplicates(TXT_FILENAME)
