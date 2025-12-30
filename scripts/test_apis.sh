#!/bin/bash
# scripts/test_apis.sh - Test API connectivity

echo "Testing Aegis Studio APIs"
echo "============================"

# Test backend health
echo -n "Backend Health: "
if curl -s http://localhost:8000/health | jq -e '.status == "healthy"' > /dev/null; then
    echo "OK"
else
    echo "FAILED"
fi

# Test model list
echo -n "Model List: "
MODEL_COUNT=$(curl -s http://localhost:8000/v1/models | jq '.data | length')
if [ "$MODEL_COUNT" -gt 0 ]; then
    echo "OK ($MODEL_COUNT models)"
else
    echo "FAILED"
fi

# Test chat completion
echo -n "Chat Completion: "
RESPONSE=$(curl -s -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "aegis-groq-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": false
    }')

if echo "$RESPONSE" | jq -e '.choices[0].message.content' > /dev/null; then
    echo "OK"
else
    echo "FAILED"
fi

echo ""
echo "Full test results in logs/test_results.json"