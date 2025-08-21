#!/bin/bash

echo "Starting all three dashboards..."

# Start Budget Dashboard on port 8052
echo "Starting Budget Dashboard on port 8052..."
cd Budget/code
python dashboard.py &
BUDGET_PID=$!
echo "Budget Dashboard started with PID: $BUDGET_PID"

# Start Decomposition Dashboard on port 8051
echo "Starting Decomposition Dashboard on port 8051..."
cd ../Decomposition/code
python dashboard.py &
DECOMP_PID=$!
echo "Decomposition Dashboard started with PID: $DECOMP_PID"

# Start Well-being Dashboard on port 8050
echo "Starting Well-being Dashboard on port 8050..."
cd ../Well-being/code
python ewbi_dashboard.py &
WELLBEING_PID=$!
echo "Well-being Dashboard started with PID: $WELLBEING_PID"

echo ""
echo "All dashboards started!"
echo "Budget Dashboard PID: $BUDGET_PID (Port 8052)"
echo "Decomposition Dashboard PID: $DECOMP_PID (Port 8051)"
echo "Well-being Dashboard PID: $WELLBEING_PID (Port 8050)"
echo ""
echo "To stop all dashboards, run: kill $BUDGET_PID $DECOMP_PID $WELLBEING_PID"
echo ""
echo "Now start ngrok with: ngrok start --all --config ngrok_config/ngrok.yml" 