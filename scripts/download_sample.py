import urllib.request
import os

def download_sample_video():
    """
    Downloads a short sample video for testing the vision pipeline.
    This provides a quick way to test the ByteTrack and YOLO integration
    without needing the full XD-Violence dataset immediately.
    """
    os.makedirs('data', exist_ok=True)
    out_path = 'data/sample.mp4'
    
    if os.path.exists(out_path):
        print(f"Sample video already exists at {out_path}")
        return
        
    # URL to a generic sample video showing people walking (creative commons)
    # Using a short, reliable test clip from a public domain source
    url = "https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4"
    
    print(f"Downloading sample video from {url}...")
    try:
        urllib.request.urlretrieve(url, out_path)
        print(f"Successfully downloaded to {out_path}")
    except Exception as e:
        print(f"Failed to download: {e}")

if __name__ == "__main__":
    download_sample_video()
