# Clever Cloud Deployment Guide

## Quick Deployment to Clever Cloud

Since you already have a Clever Cloud instance, this will be much simpler than Scaleway!

### Step 1: Prepare Your Repository

Make sure your code is in a Git repository:

```bash
# If not already a git repo
git init
git add .
git commit -m "Initial commit for Clever Cloud deployment"
```

### Step 2: Connect to Clever Cloud

#### Option A: Using Clever Cloud CLI
```bash
# Install Clever Cloud CLI
npm install -g clever-tools

# Login to Clever Cloud
clever login

# Create new application (if needed)
clever create --type python wsl-carbon-dashboard

# Or link to existing app
clever link wsl-carbon-dashboard
```

#### Option B: Using Clever Cloud Console
1. Go to [Clever Cloud Console](https://console.clever-cloud.com)
2. Create new application → Python
3. Connect your Git repository

### Step 3: Deploy

```bash
# Deploy to Clever Cloud
clever deploy

# Or push to your Git repository (if connected via console)
git push clever master
```

### Step 4: Get Your URL

After deployment, Clever Cloud will provide you with a URL like:
`https://wsl-carbon-dashboard-xxxxx.services.clever-cloud.com`

## Configuration Files

### clevercloud.json
```json
{
  "build": {
    "cache_directories": [
      "venv"
    ]
  },
  "deploy": {
    "module": "Budget.code.dashboard:asgi_app"
  },
  "hooks": {
    "pre_build": "pip install -r requirements.txt"
  },
  "environment": {
    "PYTHONPATH": "/app",
    "DASH_ENV": "production",
    "DEBUG": "False"
  },
  "scaling": {
    "min_instances": 1,
    "max_instances": 3
  }
}
```

### Procfile
```
web: uvicorn Budget.code.dashboard:asgi_app --host 0.0.0.0 --port $PORT --workers 2
```

## Integration with WSL Site

Once deployed, add this to your OSUNY site:

```html
<div class="wsl-dashboard-section">
    <div class="dashboard-header">
        <h2>Zero Carbon For All: A Fair and Inclusive Timeline</h2>
        <p class="dashboard-subtitle">
            Distributing our remaining global carbon budget fairly to stay within 1.5°C 
            implies that most developed countries have overshot their budget.
        </p>
    </div>
    
    <div class="dashboard-container">
        <iframe 
            src="https://your-dashboard.services.clever-cloud.com" 
            width="100%" 
            height="800px" 
            frameborder="0"
            loading="lazy"
            title="World Sufficiency Lab Carbon Dashboard">
        </iframe>
    </div>
    
    <div class="dashboard-footer">
        <p><small>Data sources: Global Carbon Project, World Bank, United Nations</small></p>
    </div>
</div>
```

## Advantages of Clever Cloud

✅ **Already have infrastructure** - No new setup needed
✅ **Automatic SSL** - HTTPS included
✅ **Auto-scaling** - Handles traffic spikes
✅ **Easy deployment** - Git-based deployment
✅ **Cost-effective** - Pay for what you use
✅ **Monitoring included** - Built-in logs and metrics

## Cost Estimation

- **Clever Cloud Python Instance**: ~€5-15/month (depending on usage)
- **Much cheaper than Scaleway** for this use case
- **No additional infrastructure costs**

## Monitoring

### Check Application Status
```bash
# View logs
clever logs

# Check application status
clever status

# View metrics
clever metrics
```

### Health Check
```bash
curl https://your-dashboard.services.clever-cloud.com/api/health
```

## Updates

To update the dashboard:

```bash
# Make changes to your code
git add .
git commit -m "Update dashboard data"

# Deploy
clever deploy
```

## Troubleshooting

### Common Issues:

1. **Port binding error**: Make sure `Procfile` uses `$PORT`
2. **Module not found**: Check `clevercloud.json` module path
3. **Memory issues**: Increase instance size in Clever Cloud console

### Debug Commands:
```bash
# View detailed logs
clever logs --follow

# SSH into instance (if needed)
clever ssh

# Check environment variables
clever env
```

## Next Steps

1. **Deploy to Clever Cloud** (5 minutes)
2. **Test the dashboard URL**
3. **Add iframe to WSL site**
4. **Monitor performance**

This approach is much simpler than Scaleway and leverages your existing Clever Cloud infrastructure! 