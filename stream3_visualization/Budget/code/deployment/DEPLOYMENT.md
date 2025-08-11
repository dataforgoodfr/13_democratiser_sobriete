# Budget Application Deployment Guide

## Overview
This folder contains all deployment-related files for the Budget application, which is deployed on CleverCloud.

## Files
- `clevercloud.json` - CleverCloud configuration
- `Procfile` - Process definition for CleverCloud
- `requirements.txt` - Python dependencies
- `deploy-clevercloud.sh` - Automated deployment script
- `deploy-clevercloud.md` - Detailed deployment documentation

## Quick Deploy
1. Make sure you have the CleverCloud CLI installed:
   ```bash
   npm install -g clever-tools
   clever login
   ```

2. Run the deployment script:
   ```bash
   ./deploy-clevercloud.sh
   ```

## Manual Deploy
If you prefer to deploy manually:

1. Link to your CleverCloud app:
   ```bash
   clever link YOUR_APP_NAME
   ```

2. Deploy:
   ```bash
   clever deploy
   ```

## Notes
- The application runs from the parent directory (`../dashboard:server`)
- Requirements are installed from `deployment/requirements.txt`
- The deployment script handles git initialization and app linking automatically 