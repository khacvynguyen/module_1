# curl -H "x-goog-api-key: AIzaSyD_nT97B_6M37dOEx0Ts_qsC2cZDZyhvwY" \
#   https://generativelanguage.googleapis.com/v1/models


curl -X POST \
  -H "Content-Type: application/json" \
  -H "x-goog-api-key: AIzaSyD_nT97B_6M37dOEx0Ts_qsC2cZDZyhvwY" \
  https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash-8b:generateContent \
  -d '{
    "contents": [{
      "parts": [{
        "text": "Tell me a quick fact about the moon."
      }]
    }]
  }'

