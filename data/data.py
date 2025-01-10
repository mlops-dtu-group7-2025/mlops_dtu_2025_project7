import os
import requests

def download_file(url, folder_path, filename):
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Download the file
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an error for bad responses

    # Save the file in the specified folder with the specified filename
    file_path = os.path.join(folder_path, filename)
    with open(file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def main():
    # Get the absolute path of the folder where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))  

    # Create path to the 'raw' folder in the script's directory
    raw_folder = os.path.join(script_dir, "raw")

    # URLs to download
    test_url = "https://api.zindi.africa/v1/competitions/inundata-mapping-floods-in-south-africa/files/Test.csv?auth_token=zus.v1.mWS8pNR.2d2odFAssmWV1FXtSpgkf9o9E4ZqQh"
    train_url = "https://api.zindi.africa/v1/competitions/inundata-mapping-floods-in-south-africa/files/Train.csv?auth_token=zus.v1.mWS8pNR.2d2odFAssmWV1FXtSpgkf9o9E4ZqQh"
    
    # Download the Test.csv file into the 'raw' folder
    download_file(test_url, raw_folder, "Test.csv")
    
    # Download the Train.csv file into the 'raw' folder
    download_file(train_url, raw_folder, "Train.csv")

if __name__ == "__main__":
    main()
