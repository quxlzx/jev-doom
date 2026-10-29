# Start the Doom dual-process servers on the desktop GPUs
$ErrorActionPreference = "Stop"
$exe = "D:\doomsrv\llama-server.exe"

# Spine: Qwen3-1.7B on the RTX 3070 (CUDA1), grammar-constrained
Start-Process -FilePath $exe -ArgumentList @(
  '-m','D:\doomsrv\Qwen3-1.7B-Q8_0.gguf','-ngl','99','-dev','CUDA1',
  '-c','2048','-t','8','--host','127.0.0.1','--port','8181',
  '--grammar-file','D:\doomsrv\reflex_pick.gbnf'
) -WorkingDirectory 'D:\doomsrv' -WindowStyle Hidden `
  -RedirectStandardOutput 'D:\doomsrv\spine.log' -RedirectStandardError 'D:\doomsrv\spine.err'

# Brain: Qwen3-8B on the RTX 3060 (CUDA2)
Start-Process -FilePath $exe -ArgumentList @(
  '-m','D:\doomsrv\Qwen3-8B-Q4_K_M.gguf','-ngl','99','-dev','CUDA2',
  '-c','4096','-t','8','--host','127.0.0.1','--port','8182'
) -WorkingDirectory 'D:\doomsrv' -WindowStyle Hidden `
  -RedirectStandardOutput 'D:\doomsrv\brain.log' -RedirectStandardError 'D:\doomsrv\brain.err'

Start-Sleep -Seconds 20
foreach ($p in 8181,8182) {
  try {
    $r = Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:$p/health" -TimeoutSec 5
    Write-Output "port ${p}: $($r.Content)"
  } catch { Write-Output "port ${p}: DOWN" }
}
