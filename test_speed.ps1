param([string]$url = "http://127.0.0.1:8181/completion")
$body = Get-Content D:\doomsrv\test_completion.json -Raw
$r = Invoke-WebRequest -UseBasicParsing -Uri $url -Method Post -Body $body -ContentType "application/json" -TimeoutSec 60
$j = $r.Content | ConvertFrom-Json
"content: $($j.content) | decode tok/s: $([math]::Round($j.timings.predicted_per_second,1)) | prompt tok/s: $([math]::Round($j.timings.prompt_per_second,1))"
