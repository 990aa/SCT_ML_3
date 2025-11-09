import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo, upload_file, login

# Load environment variables from .env file
load_dotenv()

# Configuration
REPO_NAME = "dogs-vs-cats-svm"  
USERNAME = "a-01a"  
MODEL_PATH = "cats-vs-dogs.keras"  # Main feature extractor model
HF_TOKEN = os.getenv("HF_TOKEN_CD")

# Files to upload (.keras format only)
FILES_TO_UPLOAD = [
    "cats-vs-dogs-components.keras",  # PCA, Scaler, and SVM components
    "README.md",
    "config.json",
]


def login_to_huggingface():
    """Login to Hugging Face using token from .env file"""
    if not HF_TOKEN:
        raise ValueError("❌ HF_TOKEN_CD not found in .env file!")
    login(token=HF_TOKEN)
    print("✓ Logged in to HuggingFace successfully!")


def check_required_files():
    """Check if all required files exist before uploading"""
    missing_files = []
    
    if not os.path.exists(MODEL_PATH):
        missing_files.append(MODEL_PATH)
    
    for file in FILES_TO_UPLOAD:
        if not os.path.exists(file):
            missing_files.append(file)
    
    return missing_files


def upload_to_huggingface(username, repo_name, model_path):
    """
    Upload model and files to Hugging Face Hub as a model repository
    """
    try:
        api = HfApi()
        repo_id = f"{username}/{repo_name}"
        
        # Create model repository
        print(f"\nCreating model repository: {repo_id}")
        try:
            create_repo(
                repo_id=repo_id,
                repo_type="model",
                exist_ok=True
            )
            print(f"✓ Repository created/verified: {repo_id}")
        except Exception as e:
            print(f"⚠ Repository might already exist: {e}")
        
        # Upload model file
        print(f"\nUploading model file: {model_path}")
        if os.path.exists(model_path):
            upload_file(
                path_or_fileobj=model_path,
                path_in_repo=os.path.basename(model_path),
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"✓ Uploaded {model_path}")
        else:
            print(f"⚠ Model file not found: {model_path}")
            return False
        
        # Upload other files
        for file in FILES_TO_UPLOAD:
            if os.path.exists(file):
                print(f"Uploading {file}...")
                upload_file(
                    path_or_fileobj=file,
                    path_in_repo=file,
                    repo_id=repo_id,
                    repo_type="model",
                )
                print(f"✓ Uploaded {file}")
            else:
                print(f"⚠ Skipping {file} (not found)")
        
        print("\n🎉 Successfully uploaded to Hugging Face!")
        print(f"🔗 View your model at: https://huggingface.co/{repo_id}")
        return True
        
    except Exception as e:
        print(f"❌ Error uploading to Hugging Face: {e}")
        return False


def main():
    """
    Main execution function - Automatically uploads to HuggingFace
    """
    print("=" * 60)
    print("Dogs vs Cats Model - Automated HuggingFace Upload")
    print("=" * 60)
    
    # Check if token exists
    if not HF_TOKEN:
        print("❌ HuggingFace token not found in .env file!")
        print("Please add HF_TOKEN_CD to your .env file")
        print("Get your token from: https://huggingface.co/settings/tokens")
        return
    
    print("✓ Found HuggingFace token in .env")
    print(f"✓ Username: {USERNAME}")
    print(f"✓ Repository: {REPO_NAME}")
    
    # Check if all required files exist
    print("\n📋 Checking for required files...")
    missing_files = check_required_files()
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure all required files exist before uploading:")
        print("   - cats-vs-dogs.keras (feature extractor)")
        print("   - cats-vs-dogs-components.keras (PCA/Scaler/SVM)")
        print("   - README.md (model card)")
        print("   - config.json (metadata)")
        return
    
    print("✓ All required files found!")
    
    try:
        # Login to HuggingFace
        print("\n🔐 Logging in to HuggingFace...")
        login_to_huggingface()
        
        # Upload to HuggingFace
        print("\n📤 Uploading to HuggingFace...")
        success = upload_to_huggingface(USERNAME, REPO_NAME, MODEL_PATH)
        
        if success:
            print("\n" + "=" * 60)
            print("🎊 UPLOAD COMPLETE!")
            print("=" * 60)
            print("\n🔗 Your model is now live at:")
            print(f"   https://huggingface.co/{USERNAME}/{REPO_NAME}")
            print("💡 Use the inference.py script to download and use the model locally")
        else:
            print("\n❌ Upload failed. Check the error messages above.")
            
    except Exception as e:
        print(f"\n❌ Error during upload process: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure HF_TOKEN_CD is set in .env file")
        print("2. Check your internet connection")
        print("3. Verify your HuggingFace username is correct")
        print("4. Make sure the token has write permissions")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
