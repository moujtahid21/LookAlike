import kagglehub
import shutil
import os


def main():
    print("⏳ Starting download via kagglehub...")

    # 1. Download latest version (Downloads to system cache)
    cache_path = kagglehub.dataset_download("yakhyokhuja/vggface2-112x112")
    print("Downloaded to cache at:", cache_path)

    # 2. Define your desired target path
    target_path = "./dataset"

    # 3. Move the files from cache to ./dataset
    print(f"📦 Moving files to {target_path}...")

    # Clean up existing folder if it exists to avoid errors
    if os.path.exists(target_path):
        shutil.rmtree(target_path)

    # Move the directory
    shutil.move(cache_path, target_path)

    print(f"✅ Success! Dataset is now located at: {os.path.abspath(target_path)}")


if __name__ == '__main__':
    main()