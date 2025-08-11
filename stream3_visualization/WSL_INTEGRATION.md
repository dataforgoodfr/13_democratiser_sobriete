# World Sufficiency Lab Dashboard Integration Guide

## Overview

This guide explains how to integrate the Carbon Budget Dashboard with the World Sufficiency Lab website on Scaleway.

## Deployment Options

### Option 1: Iframe Integration (Recommended for quick setup)

#### HTML Implementation
```html
<!-- Add this to your WSL website -->
<div class="dashboard-container">
    <h2>Zero Carbon For All: A Fair and Inclusive Timeline</h2>
    <p class="dashboard-description">
        Distributing our remaining global carbon budget fairly to stay within 1.5°C 
        implies that most developed countries have overshot their budget.
    </p>
    
    <iframe 
        src="https://your-dashboard.scaleway.com" 
        width="100%" 
        height="800px" 
        frameborder="0"
        style="border: none; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
    </iframe>
    
    <div class="dashboard-footer">
        <p><small>Data sources: Global Carbon Project, World Bank, United Nations</small></p>
    </div>
</div>
```

#### CSS Styling
```css
.dashboard-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    font-family: 'Inter', sans-serif;
}

.dashboard-description {
    color: #34495e;
    font-size: 1.1rem;
    line-height: 1.6;
    margin-bottom: 20px;
    text-align: center;
}

.dashboard-footer {
    margin-top: 20px;
    text-align: center;
    color: #7f8c8d;
}
```

### Option 2: Direct Integration (Advanced)

#### JavaScript Integration
```javascript
// Fetch dashboard data and render in WSL site
class WSLDashboard {
    constructor(baseUrl) {
        this.baseUrl = baseUrl;
        this.init();
    }
    
    async init() {
        try {
            // Check if dashboard is healthy
            const health = await this.fetchHealth();
            if (health.status === 'healthy') {
                this.loadDashboard();
            }
        } catch (error) {
            console.error('Dashboard not available:', error);
        }
    }
    
    async fetchHealth() {
        const response = await fetch(`${this.baseUrl}/api/health`);
        return await response.json();
    }
    
    async loadDashboard() {
        // Load scenarios data
        const scenarios = await fetch(`${this.baseUrl}/api/data/scenarios`);
        const scenariosData = await scenarios.json();
        
        // Load countries data
        const countries = await fetch(`${this.baseUrl}/api/data/countries`);
        const countriesData = await countries.json();
        
        // Render dashboard components
        this.renderDashboard(scenariosData, countriesData);
    }
    
    renderDashboard(scenarios, countries) {
        // Create interactive dashboard components
        const container = document.getElementById('wsl-dashboard');
        
        // Add scenario selector
        const scenarioSelect = document.createElement('select');
        scenarios.scenarios.forEach(scenario => {
            const option = document.createElement('option');
            option.value = scenario;
            option.textContent = scenario;
            scenarioSelect.appendChild(option);
        });
        
        container.appendChild(scenarioSelect);
        
        // Add iframe for the main dashboard
        const iframe = document.createElement('iframe');
        iframe.src = this.baseUrl;
        iframe.width = '100%';
        iframe.height = '800px';
        iframe.style.border = 'none';
        iframe.style.borderRadius = '8px';
        iframe.style.boxShadow = '0 4px 6px rgba(0,0,0,0.1)';
        
        container.appendChild(iframe);
    }
}

// Initialize dashboard
const dashboard = new WSLDashboard('https://your-dashboard.scaleway.com');
```

## Scaleway Deployment Steps

### 1. Prerequisites
```bash
# Install Scaleway CLI
curl -o /usr/local/bin/scw -L https://github.com/scaleway/scaleway-cli/releases/latest/download/scw-$(uname -s | tr '[:upper:]' '[:lower:]')-$(uname -m)
chmod +x /usr/local/bin/scw

# Authenticate
scw init
```

### 2. Deploy Dashboard
```bash
# Make deployment script executable
chmod +x deploy-scaleway.sh

# Run deployment
./deploy-scaleway.sh
```

### 3. Get Dashboard URL
After deployment, the script will output your dashboard URL. Save it for integration.

## Integration Examples

### WordPress Integration
```php
// Add this to your WordPress theme
function wsl_dashboard_shortcode() {
    $dashboard_url = 'https://your-dashboard.scaleway.com';
    
    return '
    <div class="wsl-dashboard-container">
        <h2>Zero Carbon For All: A Fair and Inclusive Timeline</h2>
        <iframe 
            src="' . esc_url($dashboard_url) . '" 
            width="100%" 
            height="800px" 
            frameborder="0"
            style="border: none; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        </iframe>
    </div>';
}
add_shortcode('wsl_dashboard', 'wsl_dashboard_shortcode');
```

### React Integration
```jsx
import React, { useState, useEffect } from 'react';

const WSLDashboard = ({ dashboardUrl }) => {
    const [isLoading, setIsLoading] = useState(true);
    const [isHealthy, setIsHealthy] = useState(false);
    
    useEffect(() => {
        checkHealth();
    }, []);
    
    const checkHealth = async () => {
        try {
            const response = await fetch(`${dashboardUrl}/api/health`);
            const data = await response.json();
            setIsHealthy(data.status === 'healthy');
        } catch (error) {
            console.error('Dashboard health check failed:', error);
        } finally {
            setIsLoading(false);
        }
    };
    
    if (isLoading) {
        return <div>Loading dashboard...</div>;
    }
    
    if (!isHealthy) {
        return <div>Dashboard temporarily unavailable</div>;
    }
    
    return (
        <div className="wsl-dashboard">
            <h2>Zero Carbon For All: A Fair and Inclusive Timeline</h2>
            <iframe 
                src={dashboardUrl}
                width="100%"
                height="800px"
                frameBorder="0"
                style={{
                    border: 'none',
                    borderRadius: '8px',
                    boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
                }}
                title="WSL Carbon Dashboard"
            />
        </div>
    );
};

export default WSLDashboard;
```

## Customization Options

### 1. Branding Integration
```css
/* Match WSL brand colors */
.wsl-dashboard {
    --wsl-primary: #f4d03f;
    --wsl-secondary: #2c3e50;
    --wsl-accent: #e74c3c;
}

.dashboard-header {
    background: var(--wsl-primary);
    color: var(--wsl-secondary);
    padding: 20px;
    border-radius: 8px 8px 0 0;
}
```

### 2. Responsive Design
```css
/* Mobile-friendly iframe */
@media (max-width: 768px) {
    .dashboard-container iframe {
        height: 600px;
    }
}

@media (max-width: 480px) {
    .dashboard-container iframe {
        height: 500px;
    }
}
```

### 3. Loading States
```javascript
// Add loading indicator
function showDashboardLoading() {
    return `
        <div class="dashboard-loading">
            <div class="spinner"></div>
            <p>Loading World Sufficiency Lab Dashboard...</p>
        </div>
    `;
}
```

## Monitoring and Maintenance

### Health Checks
```bash
# Check dashboard health
curl https://your-dashboard.scaleway.com/api/health

# Expected response:
# {"status": "healthy", "environment": "production"}
```

### Update Dashboard
```bash
# Redeploy with new data
./deploy-scaleway.sh
```

## Cost Estimation

- **Scaleway Container**: ~€5-10/month
- **Data Transfer**: ~€1-2/month
- **Total**: ~€6-12/month

## Security Considerations

1. **HTTPS**: Automatically provided by Scaleway
2. **CORS**: Configure if needed for direct API access
3. **Rate Limiting**: Consider adding for API endpoints
4. **Monitoring**: Set up alerts for downtime

## Support

For technical support or questions about the integration:
- Dashboard issues: Check deployment logs
- Integration issues: Review browser console
- Scaleway issues: Contact Scaleway support 