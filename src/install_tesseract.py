"""
Tesseract Language Data Downloader
===================================
Downloads required language data files for Tesseract OCR.

This script downloads the necessary .traineddata files from the
official Tesseract repository and places them in the correct location.

Author: Harry Larkin
Project: Screenshot OCR Tool (CI601)
Date: January 2026
"""

import os
import sys
import urllib.request
import ssl
from pathlib import Path


def download_file(url: str, filepath: str) -> bool:
    """
    Download a file from URL to filepath with progress.
    
    Args:
        url: Download URL
        filepath: Destination file path
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"\nDownloading: {Path(filepath).name}")
        print(f"From: {url}")
        
        # Create SSL context that doesn't verify certificates (for corporate proxies)
        context = ssl._create_unverified_context()
        
        # Download with progress
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(downloaded * 100 / total_size, 100)
                mb_downloaded = downloaded / 1024 / 1024
                mb_total = total_size / 1024 / 1024
                print(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='', flush=True)
        
        urllib.request.urlretrieve(url, filepath, reporthook=report_progress, context=context)
        print("\n  ✓ Download complete!")
        return True
        
    except Exception as e:
        print(f"\n  ✗ Download failed: {str(e)}")
        return False


def find_tessdata_directory():
    """
    Find the tessdata directory for Tesseract installation.
    
    Returns:
        Path to tessdata directory, or None if not found
    """
    # Check portable installation first
    project_root = Path(__file__).parent.parent if Path(__file__).parent.name == 'src' else Path(__file__).parent
    
    possible_locations = [
        # Portable installation in project
        project_root / "tesseract" / "tessdata",
        project_root / "src" / "tesseract" / "tessdata",
        # System installation paths
        Path(r"C:\Program Files\Tesseract-OCR\tessdata"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tessdata"),
        Path("/usr/share/tesseract-ocr/5/tessdata"),
        Path("/usr/share/tesseract-ocr/4.00/tessdata"),
        Path("/usr/local/share/tessdata"),
        Path("/opt/homebrew/share/tessdata"),
    ]
    
    for location in possible_locations:
        if location.exists():
            return location
    
    return None


def download_language_data(tessdata_dir: Path, languages: list = None):
    """
    Download language data files for Tesseract.
    
    Args:
        tessdata_dir: Path to tessdata directory
        languages: List of language codes to download (default: ['eng'])
    """
    if languages is None:
        languages = ['eng']  # Default to English only
    
    # Base URL for Tesseract trained data (fast models)
    base_url = "https://github.com/tesseract-ocr/tessdata_fast/raw/main"
    
    # List of files to download
    files_to_download = {
        'eng': f"{base_url}/eng.traineddata",  # English
        'osd': f"{base_url}/osd.traineddata",  # Orientation and script detection
    }
    
    # Add additional languages if requested
    additional_languages = {
        'fra': f"{base_url}/fra.traineddata",  # French
        'deu': f"{base_url}/deu.traineddata",  # German
        'spa': f"{base_url}/spa.traineddata",  # Spanish
        'ita': f"{base_url}/ita.traineddata",  # Italian
        'por': f"{base_url}/por.traineddata",  # Portuguese
        'rus': f"{base_url}/rus.traineddata",  # Russian
        'chi_sim': f"{base_url}/chi_sim.traineddata",  # Chinese Simplified
        'chi_tra': f"{base_url}/chi_tra.traineddata",  # Chinese Traditional
        'jpn': f"{base_url}/jpn.traineddata",  # Japanese
        'kor': f"{base_url}/kor.traineddata",  # Korean
        'ara': f"{base_url}/ara.traineddata",  # Arabic
        'hin': f"{base_url}/hin.traineddata",  # Hindi
    }
    
    print("="*70)
    print("TESSERACT LANGUAGE DATA DOWNLOADER")
    print("="*70)
    print(f"\nTarget directory: {tessdata_dir}")
    print(f"\nLanguages to download: {', '.join(languages)}")
    
    if not tessdata_dir.exists():
        print(f"\n✗ tessdata directory not found: {tessdata_dir}")
        print("Creating directory...")
        tessdata_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Directory created: {tessdata_dir}")
    
    # Always download English and OSD
    required_files = ['eng', 'osd']
    all_languages = list(set(required_files + languages))
    
    success_count = 0
    fail_count = 0
    
    for lang in all_languages:
        # Get URL
        if lang in files_to_download:
            url = files_to_download[lang]
        elif lang in additional_languages:
            url = additional_languages[lang]
        else:
            print(f"\n⚠️  Unknown language: {lang} - skipping")
            continue
        
        # Check if already exists
        output_file = tessdata_dir / f"{lang}.traineddata"
        if output_file.exists():
            print(f"\n✓ {lang}.traineddata already exists - skipping")
            success_count += 1
            continue
        
        # Download
        if download_file(url, str(output_file)):
            success_count += 1
        else:
            fail_count += 1
    
    # Summary
    print("\n" + "="*70)
    print("DOWNLOAD SUMMARY")
    print("="*70)
    print(f"✓ Successful: {success_count}")
    print(f"✗ Failed: {fail_count}")
    
    if fail_count == 0:
        print("\n✓ All language data files downloaded successfully!")
        print("\nYou can now use Tesseract OCR with these languages.")
        return True
    else:
        print(f"\n⚠️  {fail_count} downloads failed.")
        print("You may need to download manually from:")
        print("https://github.com/tesseract-ocr/tessdata_fast")
        return False


def check_existing_languages(tessdata_dir: Path):
    """
    Check what language data files already exist.
    
    Args:
        tessdata_dir: Path to tessdata directory
    """
    if not tessdata_dir.exists():
        print(f"✗ tessdata directory not found: {tessdata_dir}")
        return []
    
    traineddata_files = list(tessdata_dir.glob("*.traineddata"))
    
    if traineddata_files:
        print("\nExisting language data files:")
        for file in sorted(traineddata_files):
            size_mb = file.stat().st_size / 1024 / 1024
            print(f"  ✓ {file.name} ({size_mb:.1f} MB)")
        return [f.stem for f in traineddata_files]
    else:
        print("\n✗ No language data files found")
        return []


def main():
    """Main function to run language data downloader."""
    print("="*70)
    print("TESSERACT LANGUAGE DATA SETUP")
    print("="*70)
    
    # Find tessdata directory
    print("\nSearching for tessdata directory...")
    tessdata_dir = find_tessdata_directory()
    
    if tessdata_dir:
        print(f"✓ Found tessdata directory: {tessdata_dir}")
    else:
        print("\n✗ tessdata directory not found!")
        print("\nOptions:")
        print("1. Install Tesseract first (run install_tesseract.py)")
        print("2. Create portable installation (run setup_portable_tesseract.py)")
        print("3. Specify custom location")
        
        response = input("\nEnter option (1/2/3): ").strip()
        
        if response == '3':
            custom_path = input("Enter path to tessdata directory: ").strip()
            tessdata_dir = Path(custom_path)
            if not tessdata_dir.exists():
                print(f"\n✗ Directory not found: {tessdata_dir}")
                sys.exit(1)
        else:
            print("\nPlease set up Tesseract first, then run this script again.")
            sys.exit(1)
    
    # Check existing languages
    existing_languages = check_existing_languages(tessdata_dir)
    
    # Check if English is available
    if 'eng' in existing_languages:
        print("\n✓ English language data (eng.traineddata) found!")
        print("OCR should work for English text.")
        
        print("\nWould you like to download additional languages? (y/n): ", end='')
        response = input().strip().lower()
        
        if response != 'y':
            print("\nSetup complete - no additional languages downloaded.")
            sys.exit(0)
    else:
        print("\n⚠️  English language data (eng.traineddata) NOT found!")
        print("This is required for OCR to work.")
        print("\nDownloading required language data...")
    
    # Ask which languages to download
    print("\n" + "="*70)
    print("LANGUAGE SELECTION")
    print("="*70)
    print("\nAvailable languages:")
    print("  eng - English (required)")
    print("  fra - French")
    print("  deu - German")
    print("  spa - Spanish")
    print("  ita - Italian")
    print("  por - Portuguese")
    print("  rus - Russian")
    print("  chi_sim - Chinese Simplified")
    print("  chi_tra - Chinese Traditional")
    print("  jpn - Japanese")
    print("  kor - Korean")
    print("  ara - Arabic")
    print("  hin - Hindi")
    
    print("\nEnter language codes to download (comma-separated)")
    print("Examples: 'eng' or 'eng,fra,deu' or 'all'")
    print("Leave blank for English only: ", end='')
    
    lang_input = input().strip().lower()
    
    if not lang_input:
        languages = ['eng']
    elif lang_input == 'all':
        languages = ['eng', 'fra', 'deu', 'spa', 'ita', 'por', 'rus', 
                    'chi_sim', 'chi_tra', 'jpn', 'kor', 'ara', 'hin']
    else:
        languages = [lang.strip() for lang in lang_input.split(',')]
    
    # Download languages
    success = download_language_data(tessdata_dir, languages)
    
    if success:
        print("\n" + "="*70)
        print("SETUP COMPLETE!")
        print("="*70)
        print("\n✓ Tesseract language data installed successfully")
        print("\nYou can now use OCR with the downloaded languages.")
        print("\nTest your setup:")
        print("  python test_ocr_engine.py")
        sys.exit(0)
    else:
        print("\n⚠️  Some downloads failed")
        print("You may need to download manually from:")
        print("https://github.com/tesseract-ocr/tessdata_fast")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDownload cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)