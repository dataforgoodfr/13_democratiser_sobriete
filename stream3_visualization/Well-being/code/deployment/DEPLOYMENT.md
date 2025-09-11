# Well-being Application Deployment Guide

## Overview
This folder contains all deployment-related files for the Well-being application, which is deployed on CleverCloud.

## Files
- `clevercloud.json` - CleverCloud configuration
- `Procfile` - Process definition for CleverCloud
- `requirements.txt` - Python dependencies
- `deploy-clevercloud.sh` - Automated deployment script
- `deploy-clevercloud.md` - This documentation

## Quick Deploy
Run the deployment script:
```bash
cd deployment
./deploy-clevercloud.sh
```

## Manual Deploy Steps
1. **Commit your changes:**
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

2. **Push to GitHub (optional but recommended):**
   ```bash
   git push origin Well-being-RAw-data-and-relative-EWBI
   ```

git remote add clever-decomposition git git+ssh://git@push-n3-par-clevercloud-customers.services.clever-cloud.com/app_ac31ad44-d32f-4998-87c6-b9b699c29c63.git

3. **Deploy to CleverCloud:**
   ```bash
   git push clever-wellbeing Well-being-RAw-data-and-relative-EWBI:master
   ```

## Notes
- The application runs from the Well-being/code directory
- Requirements are installed from `deployment/requirements.txt`
- The app automatically detects the PORT environment variable from CleverCloud
- Debug mode is disabled in production (when PORT env var is set)
