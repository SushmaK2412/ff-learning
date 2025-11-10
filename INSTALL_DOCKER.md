# Docker Installation Guide

Since Docker is not currently installed on your system, follow these instructions to install it.

## macOS Installation

### Step 1: Download Docker Desktop

1. Visit the Docker Desktop download page:
   - **URL**: https://www.docker.com/products/docker-desktop/
   - Or search "Docker Desktop for Mac" in your browser

2. Click "Download for Mac"
   - Choose the version for your Mac:
     - **Apple Silicon (M1/M2/M3)**: Download "Mac with Apple Silicon"
     - **Intel Mac**: Download "Mac with Intel chip"

### Step 2: Install Docker Desktop

1. Open the downloaded `.dmg` file
2. Drag the Docker icon to your Applications folder
3. Open Docker Desktop from Applications
4. Follow the setup wizard:
   - Accept the license agreement
   - Enter your password when prompted (for system privileges)
   - Wait for Docker to start (this may take a few minutes)

### Step 3: Verify Installation

Open Terminal and run:

```bash
docker --version
# Should output: Docker version XX.XX.X, build ...

docker-compose --version
# Should output: docker-compose version X.XX.X, build ...
```

### Step 4: Start Docker Desktop

- Docker Desktop must be running to use Docker commands
- Look for the Docker whale icon in your menu bar
- If it's not running, open Docker Desktop from Applications

## System Requirements

- **macOS**: 10.15 or newer
- **RAM**: 4GB minimum (8GB recommended)
- **Disk Space**: At least 2GB free space
- **Virtualization**: Enabled (usually automatic on modern Macs)

## Troubleshooting

### Docker Desktop won't start
- Check System Preferences > Security & Privacy for any blocked applications
- Ensure you have administrator privileges
- Restart your Mac and try again

### "Docker daemon is not running"
- Make sure Docker Desktop is open and running
- Check the menu bar for the Docker whale icon
- Click the icon and select "Start" if it's stopped

### Permission denied errors
- Docker Desktop may need to be granted Full Disk Access:
  1. System Preferences > Security & Privacy > Privacy
  2. Select "Full Disk Access"
  3. Add Docker Desktop if not present

## Alternative: Run Without Docker

If you prefer not to use Docker, you can run the demo directly with Python:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download data
python data/download_data.py

# Run Streamlit app
streamlit run app.py
```

## Next Steps

Once Docker is installed:

1. **Download data** (if not done already):
   ```bash
   python data/download_data.py
   ```

2. **Build and run with Docker**:
   ```bash
   docker-compose up --build
   ```

3. **Access the dashboard**:
   - Open your browser to: http://localhost:8501

## Need Help?

- Docker Documentation: https://docs.docker.com/
- Docker Desktop for Mac: https://docs.docker.com/desktop/install/mac-install/
- Community Forums: https://forums.docker.com/

