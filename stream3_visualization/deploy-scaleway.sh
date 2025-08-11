#!/bin/bash

# Scaleway Deployment Script for World Sufficiency Lab Dashboard
# This script deploys the dashboard to Scaleway Elements

set -e

echo "🚀 Deploying World Sufficiency Lab Dashboard to Scaleway..."

# Configuration
PROJECT_NAME="wsl-carbon-dashboard"
REGION="fr-par"
INSTANCE_TYPE="DEV1-S"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if scaleway CLI is installed
if ! command -v scw &> /dev/null; then
    echo -e "${RED}❌ Scaleway CLI not found. Please install it first:${NC}"
    echo "curl -o /usr/local/bin/scw -L https://github.com/scaleway/scaleway-cli/releases/latest/download/scw-$(uname -s | tr '[:upper:]' '[:lower:]')-$(uname -m)"
    echo "chmod +x /usr/local/bin/scw"
    exit 1
fi

# Check if authenticated
if ! scw account user list &> /dev/null; then
    echo -e "${RED}❌ Not authenticated with Scaleway. Please run:${NC}"
    echo "scw init"
    exit 1
fi

echo -e "${GREEN}✅ Scaleway CLI ready${NC}"

# Create container registry if it doesn't exist
echo -e "${YELLOW}📦 Setting up container registry...${NC}"
REGISTRY_NAME="wsl-dashboard-registry"
REGISTRY_ID=$(scw registry list --output json | jq -r ".[] | select(.name == \"$REGISTRY_NAME\") | .id")

if [ -z "$REGISTRY_ID" ]; then
    echo "Creating container registry..."
    REGISTRY_ID=$(scw registry create name=$REGISTRY_NAME region=$REGION --output json | jq -r '.id')
    echo -e "${GREEN}✅ Created registry: $REGISTRY_ID${NC}"
else
    echo -e "${GREEN}✅ Using existing registry: $REGISTRY_ID${NC}"
fi

# Build and push Docker image
echo -e "${YELLOW}🔨 Building Docker image...${NC}"
IMAGE_NAME="wsl-carbon-dashboard"
TAG="latest"

# Build the image
docker build -t $IMAGE_NAME:$TAG .

# Tag for Scaleway registry
REGISTRY_URL=$(scw registry get $REGISTRY_ID --output json | jq -r '.endpoint')
FULL_IMAGE_NAME="$REGISTRY_URL/$IMAGE_NAME:$TAG"

docker tag $IMAGE_NAME:$TAG $FULL_IMAGE_NAME

# Push to registry
echo -e "${YELLOW}📤 Pushing image to Scaleway registry...${NC}"
docker push $FULL_IMAGE_NAME

echo -e "${GREEN}✅ Image pushed successfully${NC}"

# Create container instance
echo -e "${YELLOW}🚀 Creating container instance...${NC}"

# Check if instance already exists
INSTANCE_ID=$(scw container list --output json | jq -r ".[] | select(.name == \"$PROJECT_NAME\") | .id")

if [ -z "$INSTANCE_ID" ]; then
    # Create new instance
    INSTANCE_ID=$(scw container create \
        --name $PROJECT_NAME \
        --region $REGION \
        --cpu-limit 1000 \
        --memory-limit 1024 \
        --port 80 \
        --registry-image $FULL_IMAGE_NAME \
        --output json | jq -r '.id')
    
    echo -e "${GREEN}✅ Created container instance: $INSTANCE_ID${NC}"
else
    # Update existing instance
    scw container update $INSTANCE_ID --registry-image $FULL_IMAGE_NAME
    echo -e "${GREEN}✅ Updated container instance: $INSTANCE_ID${NC}"
fi

# Wait for instance to be ready
echo -e "${YELLOW}⏳ Waiting for instance to be ready...${NC}"
scw container wait $INSTANCE_ID

# Get the public URL
PUBLIC_URL=$(scw container get $INSTANCE_ID --output json | jq -r '.domain_name')

echo -e "${GREEN}🎉 Deployment successful!${NC}"
echo -e "${GREEN}🌐 Your dashboard is available at: https://$PUBLIC_URL${NC}"
echo -e "${GREEN}📊 API endpoints:${NC}"
echo -e "   Health check: https://$PUBLIC_URL/api/health"
echo -e "   Scenarios: https://$PUBLIC_URL/api/data/scenarios"
echo -e "   Countries: https://$PUBLIC_URL/api/data/countries"

# Save the URL for WSL integration
echo "https://$PUBLIC_URL" > .deployment-url

echo -e "${YELLOW}💡 For WSL website integration, use:${NC}"
echo -e "   <iframe src=\"https://$PUBLIC_URL\" width=\"100%\" height=\"800px\"></iframe>"

echo -e "${GREEN}✅ Deployment complete!${NC}" 