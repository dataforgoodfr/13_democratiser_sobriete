#!/bin/bash

# Well-being Application Deployment Script for CleverCloud

set -e

echo "🚀 Starting Well-being app deployment to CleverCloud..."

# Navigate to the deployment directory
cd "$(dirname "$0")"

# Check if git is initialized
if [ ! -d "../../../.git" ]; then
    echo "❌ Error: Not in a git repository. Please run this from the project root."
    exit 1
fi

# Check if clever-cloud remote exists
if ! git remote get-url clever-wellbeing >/dev/null 2>&1; then
    echo "❌ Error: clever-wellbeing remote not found."
    echo "Please add your CleverCloud remote:"
    echo "git remote add clever-wellbeing <your-clever-cloud-git-url>"
    exit 1
fi

echo "✅ Git repository and remote found"

# Navigate back to project root
cd ../../../

# Show current status
echo "📊 Current git status:"
git status --short

# Add all changes
echo "📦 Adding changes to git..."
git add .

# Check if there are changes to commit
if git diff --staged --quiet; then
    echo "ℹ️  No changes to commit"
else
    echo "💾 Committing changes..."
    read -p "Enter commit message (or press Enter for default): " commit_msg
    if [ -z "$commit_msg" ]; then
        commit_msg="Deploy Well-being app updates"
    fi
    git commit -m "$commit_msg"
fi

# Push to GitHub (optional)
echo "🔄 Pushing to GitHub..."
current_branch=$(git branch --show-current)
git push origin "$current_branch"

# Deploy to CleverCloud
echo "🚀 Deploying to CleverCloud..."
git push clever-wellbeing "$current_branch:master"

echo "✅ Deployment complete!"
echo "🌐 Your application should be available shortly on CleverCloud"
