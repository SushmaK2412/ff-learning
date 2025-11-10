# 🚀 Quick Start Guide

Get your federated learning demo running in 5 minutes!

## Option 1: With Docker (Recommended for Demo)

### Step 1: Install Docker
- Follow instructions in `INSTALL_DOCKER.md`
- Or visit: https://www.docker.com/products/docker-desktop/

### Step 2: Download Data
```bash
python data/download_data.py
```

### Step 3: Run Demo
```bash
docker-compose up --build
```

### Step 4: Open Dashboard
- Browser will open automatically at http://localhost:8501
- Or manually navigate to: http://localhost:8501

## Option 2: Without Docker (Local Python)

### Step 1: Setup Environment
```bash
# Run setup script
./setup.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Download Data
```bash
python data/download_data.py
```

### Step 3: Run Demo
```bash
streamlit run app.py
```

## Using the Dashboard

1. **Check Data Status** (sidebar)
   - Ensure data files are available
   - Click "Download Data" if needed

2. **Configure Training** (sidebar)
   - Select forecasting task
   - Adjust training rounds (5-20)
   - Set local epochs (3-10)

3. **Start Training**
   - Click "🚀 Start Federated Training"
   - Watch real-time progress
   - View training logs

4. **View Results**
   - Training metrics (Loss, MAPE)
   - Comparison charts
   - Data transfer analysis
   - Federated vs Centralized comparison

## Troubleshooting

### "Data files not found"
```bash
python data/download_data.py
```

### "Docker not found"
- Install Docker Desktop (see `INSTALL_DOCKER.md`)
- Or run without Docker (Option 2)

### "Port 8501 already in use"
- Change port in `docker-compose.yml`: `8502:8501`
- Or stop the process using port 8501

### Import errors
```bash
pip install -r requirements.txt
```

## Next Steps

- Try different forecasting tasks
- Adjust training parameters
- Explore the code structure
- Review the research paper in `src/`

## Need Help?

- Full documentation: See `README.md`
- Docker installation: See `INSTALL_DOCKER.md`
- Code structure: See project directories

---

**Happy Demo-ing! 🎉**

