# Deployment Guide

## Quick Sharing (Immediate)

### Using ngrok (5 minutes)
1. Sign up at [ngrok.com](https://ngrok.com) (free)
2. Get your auth token from the dashboard
3. Authenticate: `ngrok config add-authtoken YOUR_TOKEN`
4. Start dashboard: `python Budget/code/dashboard.py`
5. In another terminal: `ngrok http 8057`
6. Share the provided URL!

## Production Deployment Options

### Option 1: OVH (Recommended)
**Cost**: ~€3-5/month for basic VPS
**Setup time**: 30 minutes

#### Steps:
1. **Create OVH VPS**:
   - Go to OVH control panel
   - Create new VPS (Ubuntu 22.04)
   - Choose smallest instance (1 vCPU, 2GB RAM)

2. **Connect to server**:
   ```bash
   ssh root@YOUR_SERVER_IP
   ```

3. **Install Docker**:
   ```bash
   curl -fsSL https://get.docker.com -o get-docker.sh
   sh get-docker.sh
   apt install docker-compose
   ```

4. **Upload your code**:
   ```bash
   # On your local machine
   scp -r stream3_visualization root@YOUR_SERVER_IP:/root/
   ```

5. **Deploy**:
   ```bash
   # On the server
   cd /root/stream3_visualization
   docker-compose up -d
   ```

6. **Configure domain** (optional):
   - Point your domain to the server IP
   - Install nginx for SSL/reverse proxy

#### Cost breakdown:
- VPS: €3-5/month
- Domain (optional): €10/year
- SSL certificate: Free (Let's Encrypt)

### Option 2: Scaleway
**Cost**: Similar to OVH
**Advantage**: You already have databases there

#### Steps:
Same as OVH but:
1. Use Scaleway console to create instance
2. Choose "Development" instance type
3. Can easily connect to existing databases

### Option 3: Free Alternatives (Limited)

#### Render.com (Free tier)
- 750 hours/month free
- Automatic SSL
- Git-based deployment

#### Railway.app
- $5 credit/month free
- Easy deployment
- Good for prototypes

## Security Considerations

### Basic Security:
1. **Environment variables** for sensitive data
2. **Firewall rules** (only allow ports 80, 443, 22)
3. **Regular updates**
4. **Backup strategy**

### For production:
1. **SSL certificate** (mandatory)
2. **Rate limiting**
3. **User authentication** if needed
4. **Monitoring/logging**

## Monitoring

### Basic health check:
```bash
curl -f http://your-domain.com
```

### Advanced monitoring:
- Uptime monitoring (UptimeRobot - free)
- Error tracking (Sentry - free tier)
- Performance monitoring (built into most VPS providers)

## Maintenance

### Updates:
```bash
# On server
cd /root/stream3_visualization
git pull  # if using git
docker-compose down
docker-compose build
docker-compose up -d
```

### Backups:
- Data files are read-only, so minimal backup needed
- Consider backing up the entire application folder weekly

## Recommended Approach

1. **Start with ngrok** for immediate sharing
2. **Deploy to OVH** for production (you have existing contract)
3. **Add domain and SSL** for professional look
4. **Set up basic monitoring**

Total monthly cost: €3-8 depending on instance size.