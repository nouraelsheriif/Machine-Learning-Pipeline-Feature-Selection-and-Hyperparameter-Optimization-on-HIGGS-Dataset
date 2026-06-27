import gzip
import shutil
import os

print("Extracting HIGGS dataset...")

# Extract the gzip file
try:
    with gzip.open('HIGGS.csv.gz', 'rb') as f_in:
        with open('HIGGS.csv', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    print(" Extraction complete! File saved as HIGGS.csv")
    
    # Check file size
    if os.path.exists('HIGGS.csv'):
        size = os.path.getsize('HIGGS.csv') / (1024 * 1024)
        print(f" File size: {size:.2f} MB")
    else:
        print(" Error: File was not created!")
        
except FileNotFoundError:
    print(" Error: HIGGS.csv.gz not found in the current directory!")
    print(" Current directory contents:")
    os.system('ls -la')
except Exception as e:
    print(f" Error: {e}")