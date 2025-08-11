#!/bin/bash

# Clever Cloud Deployment Script for WSL Dashboard
# This script deploys the dashboard to Clever Cloud

set -e

echo "🚀 Deploying World Sufficiency Lab Dashboard to Clever Cloud..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if clever-tools is installed
if ! command -v clever &> /dev/null; then
    echo -e "${RED}❌ Clever Cloud CLI not found. Please install it first:${NC}"
    echo "npm install -g clever-tools"
    echo "clever login"
    exit 1
fi

# Check if authenticated
if ! clever whoami &> /dev/null; then
    echo -e "${RED}❌ Not authenticated with Clever Cloud. Please run:${NC}"
    echo "clever login"
    exit 1
fi

echo -e "${GREEN}✅ Clever Cloud CLI ready${NC}"

# Check if git repository is initialized (go up one level to code directory)
if [ ! -d "../.git" ]; then
    echo -e "${YELLOW}📦 Initializing git repository...${NC}"
    cd ..
    git init
    git add .
    git commit -m "Initial commit for Clever Cloud deployment"
    cd deployment
fi

# Check if app is linked
if ! clever link &> /dev/null; then
    echo -e "${YELLOW}🔗 Linking to Clever Cloud app...${NC}"
    echo "Please enter your Clever Cloud app name:"
    read APP_NAME
    
    if [ -z "$APP_NAME" ]; then
        echo -e "${YELLOW}Creating new app: wsl-carbon-dashboard${NC}"
        clever create --type python wsl-carbon-dashboard
        clever link wsl-carbon-dashboard
    else
        clever link $APP_NAME
    fi
else
    echo -e "${GREEN}✅ Already linked to Clever Cloud app${NC}"
fi

# Deploy to Clever Cloud
echo -e "${YELLOW}🚀 Deploying to Clever Cloud...${NC}"
clever deploy

# Get the app URL
APP_URL=$(clever info | grep "URL:" | awk '{print $2}')

if [ -n "$APP_URL" ]; then
    echo -e "${GREEN}🎉 Deployment successful!${NC}"
    echo -e "${GREEN}🌐 Your dashboard is available at: $APP_URL${NC}"
    echo -e "${GREEN}📊 API endpoints:${NC}"
    echo -e "   Health check: $APP_URL/api/health"
    echo -e "   Scenarios: $APP_URL/api/data/scenarios"
    echo -e "   Countries: $APP_URL/api/data/countries"
    
    # Save the URL for WSL integration
    echo "$APP_URL" > .deployment-url
    
    echo -e "${YELLOW}💡 For WSL website integration, use:${NC}"
    echo -e "   <iframe src=\"$APP_URL\" width=\"100%\" height=\"800px\"></iframe>"
    
    echo -e "${GREEN}✅ Deployment complete!${NC}"
else
    echo -e "${RED}❌ Could not get app URL. Please check Clever Cloud console.${NC}"
fi 